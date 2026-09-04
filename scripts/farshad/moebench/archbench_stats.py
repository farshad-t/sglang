"""Fetch archbench expert-stats CSVs straight out of the archbench git repo.

The benchmark must read the SAME stats bytes the projection reads, and the local
checkouts on this box are not a trustworthy source of them: the qwen35 tree here is
a git worktree whose `.git` points at another user's home directory, so it cannot
be updated, verified, or attributed to a revision. So resolve the files from the
repo instead, and record the commit sha alongside every measurement -- a measured
number that cannot be tied to a stats revision cannot be compared to a projection
run months later.

Fetching is cheap and incremental: a blobless shallow fetch of the one branch
(~1 s), then a lazy per-file blob fetch. Results are cached under
~/.cache/moebench/archbench/ keyed by commit sha, so re-runs on a pinned sha do
no network I/O at all.

Offline / air-gapped boxes have two escapes that need no network:
  --ab-local-repo DIR   read blobs from an existing clone or worktree of the repo
  --stats-dir DIR       read plain CSV files from a directory
"""

import json
import os
import re
import subprocess
from typing import Dict, Optional, Tuple

AB_REPO = "https://github.com/intel-restricted/frameworks.ai.benchmarking.archbench-1.git"
AB_REF = "farshad/llama4-qwen3-qwen35"
MAPPING_PATH = "tools/cpu/config/qwen35/expert_stats_mapping.json"

CACHE_ROOT = os.environ.get(
    "MOEBENCH_CACHE", os.path.expanduser("~/.cache/moebench/archbench"))

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(*args: str, cwd: Optional[str] = None, binary: bool = False):
    # safe.directory=* because the cache is routinely reached from a container that
    # runs as a different uid than the one that created it; every repo we touch here
    # is either our own cache or a path the user pointed us at explicitly.
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never block on a credential prompt
    r = subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("git {} failed ({}): {}".format(
            " ".join(args), r.returncode, r.stderr.decode(errors="replace").strip()))
    return r.stdout if binary else r.stdout.decode()


class StatsSource:
    """Resolves repo-relative paths to local files. One of three backends."""

    def __init__(self, repo: str = AB_REPO, ref: str = AB_REF,
                 commit: Optional[str] = None, local_repo: Optional[str] = None,
                 offline: bool = False):
        self.repo = repo
        self.ref = ref
        self.local_repo = local_repo
        self.offline = offline
        if local_repo:
            # A clone or worktree already on disk: resolve the ref there, no network.
            self.commit = commit or _git("rev-parse", self._local_rev(ref),
                                         cwd=local_repo).strip()
            self.origin = f"{local_repo} @ {ref}"
        else:
            self.commit = commit or self._resolve_remote(offline)
            self.origin = f"{repo} @ {ref}"
        self.files_dir = os.path.join(CACHE_ROOT, "files", self.commit)

    def _local_rev(self, ref: str) -> str:
        """Prefer the remote-tracking ref: a local branch of the same name may be
        checked out at an older commit than what was last fetched."""
        for cand in (f"origin/{ref}", ref):
            try:
                _git("rev-parse", "--verify", "--quiet", cand + "^{commit}",
                     cwd=self.local_repo)
                return cand
            except RuntimeError:
                continue
        raise SystemExit(f"{self.local_repo}: no such ref {ref!r} (tried origin/{ref}, {ref})")

    def _git_cache(self) -> str:
        """A private blobless shallow clone, created on first use and reused."""
        d = os.path.join(CACHE_ROOT, "git")
        if not os.path.isdir(os.path.join(d, ".git")):
            os.makedirs(d, exist_ok=True)
            _git("init", "-q", d)
            _git("remote", "add", "origin", self.repo, cwd=d)
        else:
            # Point at the requested repo in case it was overridden.
            _git("remote", "set-url", "origin", self.repo, cwd=d)
        return d

    def _ref_note(self) -> str:
        """Where the last resolved sha for this (repo, ref) is remembered, so an
        offline run can still turn a branch NAME into a commit."""
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", f"{self.repo}#{self.ref}")
        return os.path.join(CACHE_ROOT, "refs", slug)

    def _resolve_remote(self, offline: bool) -> str:
        note = self._ref_note()
        if offline:
            if os.path.exists(note):
                with open(note) as f:
                    sha = f.read().strip()
                print(f"[archbench_stats] offline: {self.ref} -> {sha} (last resolved "
                      f"on this box; may be behind the branch)")
                return sha
            raise SystemExit(
                f"--ab-offline cannot resolve {self.ref!r} to a commit: this box has "
                f"never resolved it online. Pass --ab-commit <sha>, or use "
                f"--ab-local-repo / --stats-dir.")
        out = _git("ls-remote", self.repo, f"refs/heads/{self.ref}").strip()
        if not out:
            raise SystemExit(f"{self.repo}: no branch {self.ref!r}")
        sha = out.split()[0]
        os.makedirs(os.path.dirname(note), exist_ok=True)
        with open(note, "w") as f:
            f.write(sha + "\n")
        return sha

    def _have_commit(self, d: str) -> bool:
        try:
            _git("cat-file", "-e", self.commit + "^{commit}", cwd=d)
            return True
        except RuntimeError:
            return False

    def read(self, repo_path: str) -> bytes:
        if self.local_repo:
            return _git("cat-file", "blob", f"{self.commit}:{repo_path}",
                        cwd=self.local_repo, binary=True)
        d = self._git_cache()
        if not self._have_commit(d):
            _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin",
                 f"refs/heads/{self.ref}", cwd=d)
            head = _git("rev-parse", "FETCH_HEAD", cwd=d).strip()
            if head != self.commit:
                # The pinned sha is not the branch tip; ask for it by name. GitHub
                # allows fetching a reachable sha directly.
                _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin",
                     self.commit, cwd=d)
            # Anchor it so gc cannot drop the cached objects between runs.
            _git("update-ref", f"refs/moebench/{self.commit}", self.commit, cwd=d)
        return _git("cat-file", "blob", f"{self.commit}:{repo_path}", cwd=d, binary=True)

    def materialize(self, repo_path: str) -> str:
        """Write a repo file into the sha-keyed cache and return its local path."""
        dst = os.path.join(self.files_dir, repo_path)
        if os.path.exists(dst):
            return dst
        if self.offline:
            # A blobless clone fetches blobs lazily, so a cache miss here would send
            # git to the network -- which on an isolated box (or inside a container)
            # means a multi-minute connect timeout and then a confusing promisor
            # error. Fail immediately with the fix instead.
            raise SystemExit(
                f"--ab-offline: {repo_path} is not in the cache for commit "
                f"{self.commit[:12]}.\nWarm it on a host that can reach the repo:\n"
                f"  python3 bench_moe_cpu.py --ab-prefetch [--expert-stats-mode ...]\n"
                f"cache root: {CACHE_ROOT}")
        blob = self.read(repo_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, dst)  # atomic, so a killed run cannot leave a half file
        return dst

    def mapping(self, path: str = MAPPING_PATH) -> Dict:
        # Via materialize(), so an offline run whose cache already holds the mapping
        # never reaches git at all.
        with open(self.materialize(path)) as f:
            return json.load(f)

    def resolve_stats(self, model_key: str, dtype: str, stats_mode: str,
                      mapping_path: str = MAPPING_PATH) -> Tuple[str, str, str]:
        """Look up the (decode, prefill) CSVs exactly the way archbench does:
        expert_stats_mapping.json keyed by "<hidden>-<num_experts>-<moe_inter>",
        then dtype, then per_layer / layers_averaged. Returns
        (decode_repo_path, prefill_repo_path, model_name)."""
        m = self.mapping(mapping_path)
        if model_key not in m:
            raise SystemExit(f"expert_stats_mapping has no model_key {model_key!r}; "
                             f"available: {sorted(m)}")
        block = m[model_key]
        if dtype not in block:
            avail = [k for k in block if k != "model_name"]
            raise SystemExit(f"model {block.get('model_name', model_key)} has no dtype "
                             f"{dtype!r}; available: {avail}")
        # Legacy mappings put decode/prefill straight under the dtype.
        mode_block = block[dtype].get(stats_mode, block[dtype])
        return mode_block["decode"], mode_block["prefill"], block.get("model_name", model_key)

    def prefetch(self, model_key: Optional[str] = None,
                 mapping_path: str = MAPPING_PATH) -> int:
        """Pull every stats CSV the mapping references into the cache, so a later run
        on an isolated box (or inside a container) needs no network at all. Restrict
        to one model_key, or pass None for all of them."""
        m = self.mapping(mapping_path)
        paths = set()
        for key, block in m.items():
            if model_key and key != model_key:
                continue
            for dtype, dblock in block.items():
                if not isinstance(dblock, dict):
                    continue  # "model_name"
                for mode, mblock in dblock.items():
                    entries = mblock if isinstance(mblock, dict) else dblock
                    for phase in ("decode", "prefill"):
                        if isinstance(entries, dict) and isinstance(
                                entries.get(phase), str):
                            paths.add(entries[phase])
        for p in sorted(paths):
            print(f"  {self.materialize(p)}")
        return len(paths)


def add_args(p) -> None:
    g = p.add_argument_group("expert-stats source (archbench repo)")
    g.add_argument("--ab-repo", default=AB_REPO, help="archbench git remote")
    g.add_argument("--ab-ref", default=AB_REF, help=f"branch to read stats from "
                                                    f"(default {AB_REF})")
    g.add_argument("--ab-commit", default=None,
                   help="pin an exact stats commit sha (skips ls-remote; makes a run "
                        "reproducible against a fixed stats revision)")
    g.add_argument("--ab-local-repo", default=None,
                   help="read blobs from this existing clone/worktree instead of the "
                        "network")
    g.add_argument("--ab-offline", action="store_true",
                   help="never touch the network (requires --ab-commit already cached)")
    g.add_argument("--stats-dir", default=None,
                   help="bypass git entirely: a directory holding the CSVs by basename")
    g.add_argument("--ab-prefetch", action="store_true",
                   help="cache every stats CSV the mapping references, then exit. Run "
                        "this once on a networked host so --ab-offline works afterwards "
                        "(e.g. inside a container with no network).")


def from_args(args) -> Optional[StatsSource]:
    if args.stats_dir:
        return None
    if args.ab_commit and not _SHA_RE.match(args.ab_commit):
        raise SystemExit(f"--ab-commit must be a full 40-hex sha, got {args.ab_commit!r}")
    return StatsSource(repo=args.ab_repo, ref=args.ab_ref, commit=args.ab_commit,
                       local_repo=args.ab_local_repo, offline=args.ab_offline)
