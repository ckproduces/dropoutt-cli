"""A vectorised reading of the py3langid classifier.

py3langid spends nearly all of its time in one Python loop. ``instance2fv``
steps a deterministic automaton one byte at a time — a dict lookup and a list
extend per byte — and a scan calls it once per record. On a real multilingual
corpus that loop was the single largest item in the whole scan pass, larger
than deduplication and larger than every content check together.

The loop is not vectorisable as written: the state after byte *i* depends on
the state after byte *i-1*. What is vectorisable is **what the loop computes**.

The automaton is an Aho-Corasick machine over byte n-grams. After reading
``text[:i+1]`` it sits in the state naming the longest suffix of that prefix
which is a node of the pattern trie, and the state's output set is every
pattern that ends at position *i*. So the feature vector the loop accumulates
is, exactly:

    fv[f] = the number of times the byte string of feature f occurs in the text

which is a bag of n-grams — countable with array arithmetic over the whole
batch at once, in any order, with no automaton at all.

:func:`patterns_from_automaton` recovers the byte string of every feature from
the transition table, and :func:`verify_reconstruction` proves the recovery is
exact for the model actually installed. The proof is not a test fixture: it
runs at load, and :class:`NgramModel` refuses to build when it fails, so the
fast path is only ever used on an automaton it has checked. See
``docs/research.md`` for the measurements.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

#: Longest n-gram this module will index. The shipped langid model tops out at
#: four bytes; the packing below keeps a whole n-gram in one uint32, which is
#: what makes the lookups a single gather rather than a comparison chain.
MAX_NGRAM = 4

#: Records per vectorised classification. Larger batches measured no faster —
#: the work is linear in bytes and the fixed costs are already amortised by a
#: few dozen records — and cost linearly more transient memory; smaller ones
#: start losing the amortisation that is the entire point.
BATCH = 256

_U32_8 = np.uint32(8)
_U64_8 = np.uint64(8)


@dataclass(frozen=True)
class _Level:
    """The lookup for n-grams of one length.

    ``table`` is a direct index for one- and two-byte grams, where the whole key
    space is 256 or 65,536 entries and a gather beats any search. ``keys`` and
    ``values`` are the sorted fallback for longer grams, where the key space is
    too large to materialise.
    """

    n: int
    table: np.ndarray | None
    keys: np.ndarray
    values: np.ndarray


def _transition_table(tk_nextmove) -> np.ndarray:
    """The automaton's transitions as a ``(states, 256)`` array."""
    flat = np.asarray(tk_nextmove)
    if flat.ndim != 1 or flat.size % 256:
        raise ValueError("transition table is not a flat states-by-256 array")
    return flat.reshape(-1, 256)


def state_strings(moves: np.ndarray) -> list[bytes]:
    """The byte string that reaches each state, by breadth-first search.

    Every state of an Aho-Corasick DFA *is* a node of the pattern trie, and the
    string of a trie node at depth *d* is reachable in exactly *d* steps and no
    fewer — a shorter path would end in a state whose string is shorter. So
    level-order traversal from the root discovers each state at a depth equal to
    the length of its string, and the path taken to discover it spells that
    string out.
    """
    total = moves.shape[0]
    strings: list[bytes] = [b""] * total
    seen = bytearray(total)
    seen[0] = 1
    queue = deque([0])
    rows = moves.tolist()
    while queue:
        state = queue.popleft()
        prefix = strings[state]
        row = rows[state]
        for byte in range(256):
            nxt = row[byte]
            if not seen[nxt]:
                seen[nxt] = 1
                strings[nxt] = prefix + bytes((byte,))
                queue.append(nxt)
    if not all(seen):
        raise ValueError("transition table has unreachable states")
    return strings


def patterns_from_automaton(moves: np.ndarray, outputs, num_features: int) -> list[bytes]:
    """The byte string of every feature index.

    A feature is emitted by a state exactly when its pattern is a suffix of that
    state's string, and the pattern is itself a trie node, so the shortest
    string among the states that emit a feature *is* that feature's pattern.
    """
    strings = state_strings(moves)
    found: list[bytes | None] = [None] * num_features
    for state, emitted in outputs.items():
        text = strings[state]
        for feature in emitted:
            current = found[feature]
            if current is None or len(text) < len(current):
                found[feature] = text
    missing = [i for i, p in enumerate(found) if p is None]
    if missing:
        raise ValueError(f"{len(missing)} features are emitted by no state")
    return [p for p in found if p is not None]


def verify_reconstruction(moves: np.ndarray, outputs, patterns: list[bytes]) -> None:
    """Prove the recovered patterns reproduce the automaton, or raise.

    Exhaustive rather than sampled, and in two halves that together close the
    induction the module docstring appeals to.

    **Outputs.** For every state, the output set must equal the set of patterns
    that are suffixes of that state's string.

    **Transitions.** For every state and every byte, the destination's string
    must be the longest suffix of ``string(state) + byte`` that is itself a
    state — which is the defining property of an Aho-Corasick automaton, and
    the half a table can silently lack. Checking outputs alone accepts a table
    whose failure links are wrong: the per-state sets still match, but on real
    input the walk visits different states than the suffix structure implies
    and emits different counts, so the bag-of-n-grams shortcut would disagree
    with the loop it replaces without anything refusing to build.

    Together they give the guarantee by induction on input position: the state
    after any prefix is the longest trie-suffix of it (transitions), and what
    is emitted there is exactly every pattern ending at that position
    (outputs). Costs about a quarter of a second, once per process, and is
    what lets this module be used instead of the loop it replaces rather than
    alongside it.
    """
    strings = state_strings(moves)
    longest = max(len(p) for p in patterns)
    index = {p: i for i, p in enumerate(patterns)}
    if len(index) != len(patterns):
        raise ValueError("two features share a pattern")
    for state, text in enumerate(strings):
        expected = set()
        for size in range(1, min(longest, len(text)) + 1):
            feature = index.get(text[len(text) - size :])
            if feature is not None:
                expected.add(feature)
        if expected != set(outputs.get(state, ())):
            raise ValueError(f"state {state} does not match its reconstructed patterns")
    _verify_transitions(moves, strings)


def _verify_transitions(moves: np.ndarray, strings: list[bytes]) -> None:
    """Every transition must land on the longest suffix that is a state.

    Vectorised because the table is states x 256 — 2.3 million transitions for
    the shipped model, which a Python loop would spend seconds on. States are
    grouped by string length; each group's strings are packed into uint64 lanes,
    every possible next byte is appended in one broadcast, and each candidate
    suffix length is resolved with one ``searchsorted`` against the states of
    that length. Suffix keys are compared within a single length, so a string
    with leading zero bytes cannot collide with a shorter one.
    """
    depth = max(len(s) for s in strings)
    if len(set(strings)) != len(strings):
        raise ValueError("two states share a string")

    by_length: dict[int, list[int]] = {}
    for state, text in enumerate(strings):
        by_length.setdefault(len(text), []).append(state)

    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for length, members in by_length.items():
        if length == 0:
            continue
        pairs = sorted(
            (int.from_bytes(strings[state], "big"), state) for state in members
        )
        tables[length] = (
            np.array([key for key, _ in pairs], dtype=np.uint64),
            np.array([state for _, state in pairs], dtype=np.int64),
        )

    every_byte = np.arange(256, dtype=np.uint64)
    for length, members in by_length.items():
        idx = np.asarray(members, dtype=np.int64)
        packed = np.array(
            [int.from_bytes(strings[state], "big") for state in members],
            dtype=np.uint64,
        )
        appended = (packed[:, None] << _U64_8) | every_byte[None, :]
        # Longest match wins: start at the root and let each longer suffix
        # length overwrite where it hits.
        expected = np.zeros(appended.shape, dtype=np.int64)
        for size in range(1, min(length + 1, depth) + 1):
            entry = tables.get(size)
            if entry is None:
                continue
            keys, states = entry
            suffix = (appended & np.uint64((1 << (8 * size)) - 1)).ravel()
            where = np.searchsorted(keys, suffix)
            np.clip(where, 0, keys.size - 1, out=where)
            hit = keys[where] == suffix
            found = np.where(hit, states[where], -1).reshape(appended.shape)
            expected = np.where(found >= 0, found, expected)
        if not np.array_equal(moves[idx], expected):
            raise ValueError(
                "transition table disagrees with the longest-suffix structure"
            )


def _levels(patterns: list[bytes]) -> list[_Level]:
    by_length: dict[int, list[tuple[int, int]]] = {}
    for feature, pattern in enumerate(patterns):
        packed = int.from_bytes(pattern, "big")
        by_length.setdefault(len(pattern), []).append((packed, feature))
    levels: list[_Level] = []
    for n in sorted(by_length):
        pairs = sorted(by_length[n])
        keys = np.array([k for k, _ in pairs], dtype=np.uint32)
        values = np.array([f for _, f in pairs], dtype=np.int32)
        table = None
        if n <= 2:
            table = np.full(1 << (8 * n), -1, dtype=np.int32)
            table[keys.astype(np.int64)] = values
        levels.append(_Level(n=n, table=table, keys=keys, values=values))
    return levels


class NgramModel:
    """The py3langid model, classified in batches.

    Produces the same label as ``LanguageIdentifier.classify`` and a probability
    that agrees with it to within float32 summation error — the arithmetic is
    the same dot product against the same table, gathered in a different order.
    Measured on 9,000 real multilingual documents: no label differs, and the
    largest probability difference is 6e-5.
    """

    def __init__(self, identifier) -> None:
        # The classifier itself needs scipy only inside log_probs. Importing it
        # here makes a missing scipy fail construction instead, which is what
        # lets langid fall back to the library identifier: an instance that
        # builds fine and then raises on every batch would be swallowed into
        # "unknown" for the whole corpus.
        from scipy.sparse import coo_matrix  # noqa: F401

        moves = _transition_table(identifier.tk_nextmove)
        patterns = patterns_from_automaton(
            moves, identifier.tk_output, identifier.nb_numfeats
        )
        if max(len(p) for p in patterns) > MAX_NGRAM:
            raise ValueError("model uses n-grams longer than this module packs")
        verify_reconstruction(moves, identifier.tk_output, patterns)
        levels = _levels(patterns)
        self._by_length = {level.n: level for level in levels}
        self._max_ngram = max(self._by_length)
        self._ptc = np.ascontiguousarray(identifier.nb_ptc)
        self._pc = np.asarray(identifier.nb_pc)
        self.classes: list[str] = list(identifier.nb_classes)
        self.n_features = int(identifier.nb_numfeats)

    # -- feature counting --------------------------------------------------

    def _pairs(self, buffers: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
        """``(record index, feature index)`` for every n-gram occurrence."""
        lengths = np.fromiter(
            (len(b) for b in buffers), dtype=np.int64, count=len(buffers)
        )
        data = np.frombuffer(b"".join(buffers), dtype=np.uint8)
        owner = np.repeat(np.arange(len(buffers), dtype=np.int64), lengths)
        size = data.size
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        packed = data.astype(np.uint32)
        for n in range(1, self._max_ngram + 1):
            span = size - n + 1
            if span <= 0:
                break
            if n > 1:
                # Roll the previous length's keys forward one byte rather than
                # rebuilding them: the n-gram at i is the (n-1)-gram at i with
                # one more byte on the end.
                packed = (packed[:span] << _U32_8) | data[n - 1 : n - 1 + span]
            level = self._by_length.get(n)
            if level is None:
                continue
            if level.table is not None:
                feature = level.table[packed[:span]]
                hit = feature >= 0
            else:
                where = np.searchsorted(level.keys, packed[:span])
                np.clip(where, 0, level.keys.size - 1, out=where)
                hit = level.keys[where] == packed[:span]
                feature = level.values[where]
            if n > 1:
                # An n-gram that straddles two records belongs to neither.
                hit &= owner[:span] == owner[n - 1 : n - 1 + span]
            picked = np.flatnonzero(hit)
            if picked.size:
                rows.append(owner[picked])
                cols.append(feature[picked])
        if not rows:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(rows), np.concatenate(cols).astype(np.int64)

    def counts(self, buffers: list[bytes]) -> np.ndarray:
        """The dense ``(records, features)`` count matrix for one batch.

        Not used on the scan path — :meth:`log_probs` never materialises this —
        but it is the definition ``instance2fv`` computes one record at a time,
        and the equivalence tests compare against it directly.
        """
        rows, cols = self._pairs(buffers)
        total = len(buffers) * self.n_features
        flat = np.bincount(rows * self.n_features + cols, minlength=total)
        return flat.reshape(len(buffers), self.n_features)

    # -- classification ----------------------------------------------------

    def log_probs(self, buffers: list[bytes]) -> np.ndarray:
        """Unnormalised per-class log probability, one row per record.

        The multiply is a sparse one, and that is not a performance choice — a
        dense ``(records, features) @ (features, classes)`` is a BLAS ``gemm``,
        and on macOS numpy links against Accelerate, whose ``gemm`` dispatches
        through libdispatch. **A process that has called it cannot be forked and
        then call it again**: the child segfaults on its first multiply. The
        scan pass runs in forked workers, so nothing on it may reach a ``gemm``.
        ``gemv``, SciPy's sparse multiply and ``einsum`` are all unaffected;
        ``tests/test_fork_safety.py`` holds the line.

        The sparse form is the honest shape anyway. Six or seven hundred n-gram
        occurrences land in a row that is 7,480 wide, so the dense matrix is
        mostly zeros being multiplied by weights. Measured on the corpus in
        ``docs/research.md`` the two produce identical numbers, and the sparse
        one is a third of the arithmetic.
        """
        from scipy.sparse import coo_matrix

        rows, cols = self._pairs(buffers)
        # An occurrence is a one; repeated occurrences of the same n-gram in the
        # same record are duplicate entries, which the conversion to CSR sums.
        # That sum is the count `instance2fv` builds, arrived at without ever
        # writing the count down. py3langid holds it as uint16 and wraps above
        # 65,535; no count can exceed the record's own length in bytes, so for
        # any record shorter than that the two are the same number.
        occurrences = coo_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)),
            shape=(len(buffers), self.n_features),
        ).tocsr()
        return occurrences @ self._ptc + self._pc

    def classify_many(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        """Label and probability for each text, in batches."""
        labels: list[str] = []
        scores = np.zeros(len(texts), dtype=np.float64)
        names = self.classes
        for start in range(0, len(texts), BATCH):
            chunk = texts[start : start + BATCH]
            buffers = [t.encode("utf-8", "surrogatepass") for t in chunk]
            pd = self.log_probs(buffers)
            best = pd.argmax(axis=1)
            # py3langid renormalises the whole log-probability vector into a
            # distribution and then takes the winner's entry. Only the winner's
            # entry is ever read, and that entry is 1 / sum_j exp(pd_j - pd_max)
            # — the same sum over the same values in the same order, so this is
            # the same float, computed once per record instead of C times.
            with np.errstate(over="ignore"):
                probability = 1.0 / np.exp(pd - pd[np.arange(len(chunk)), best][:, None]).sum(
                    axis=1
                )
            labels.extend(names[int(i)] for i in best)
            scores[start : start + len(chunk)] = probability
        return labels, scores

    def classify(self, text: str) -> tuple[str, float]:
        labels, scores = self.classify_many([text])
        return labels[0], float(scores[0])
