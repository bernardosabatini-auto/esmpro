# Transfer to the H200 cluster

Two parts: this bundle (106 MB) and two large data files (33.5 GB) kept outside
it, so the Spark does not have to hold a second 33 GB copy.

Set these once, editing both:

```bash
REMOTE=user@h200.example.edu
DEST=/scratch/$USER/esm_proae          # needs ~40 GB free
SRC=/home/guest/projects/esm_proae
```

## 1. The bundle

```bash
ssh $REMOTE "mkdir -p $DEST/data/phase1_dataset"
scp -r $SRC/h200_bundle/* $REMOTE:$DEST/
```

## 2. The data — 33.5 GB, use rsync so it resumes

```bash
rsync -avP --partial \
  $SRC/data/phase1_dataset/dataset_100k.h5 \
  $SRC/data/phase1_dataset/backbone_100k.h5 \
  $REMOTE:$DEST/data/phase1_dataset/
```

`rsync -P` resumes a broken transfer. Plain `scp` restarts from zero, which on
33 GB is worth avoiding.

`dataset_100k.h5` (32.7 GB) is required. `backbone_100k.h5` (797 MB) is only
needed for true-FAPE frames, which are deliberately off — see CLAUDE.md
section 8 — so you may defer it.

## 3. Verify

Check sizes match exactly. A truncated HDF5 often opens fine and fails later.

```bash
# on the Spark
md5sum $SRC/data/phase1_dataset/dataset_100k.h5

# on the cluster
md5sum $DEST/data/phase1_dataset/dataset_100k.h5
```

The bundle also ships `checksums.md5` covering the code and both checkpoints:

```bash
cd $DEST && md5sum -c checksums.md5
```

## 4. One command that proves the whole transfer

```bash
export ESM_PROAE_ROOT=$DEST
export FOLDSEEK_BIN=$(which foldseek)
cd $ESM_PROAE_ROOT/code
python gate6_score_checkpoint.py --ckpt smallscale_final_40k.pt --n 100 --bs 10
```

Expect **TM ≈ 0.42-0.44, TM>0.5 ≈ 0.32, RMSD ≈ 11.5 A, coverage 100/100**.

This exercises the dataset, the decoder checkpoint, the head checkpoint,
Foldseek and the path wiring at once. If it passes, the transfer is good.
Coverage below 100/100 means Foldseek's prefilter is back — see CLAUDE.md bug 2.

## 5. Hand Claude the instructions

`CLAUDE.md` sits at the bundle root and is written for the Claude instance on
that machine. It opens with the three silent measurement bugs, because each one
has already cost this project multiple days.

## Expected layout on the cluster

```
$DEST/
  CLAUDE.md
  TRANSFER.md
  checksums.md5
  code/                      7 scripts, portable via ESM_PROAE_ROOT
  ProteinAE_v1/              decoder package + ae_r1_d8_v1.ckpt
  data/phase1_dataset/
    dataset_100k.h5          32.7 GB   (rsync)
    backbone_100k.h5         797 MB    (rsync, optional for now)
    smallscale_final_40k.pt  33 MB     (in the bundle)
  docs/
```

---

# Direction B — pull from the cluster (Spark is the remote)

Use this when you are logged into the H200 and it can reach the Spark. Nothing
about the bundle changes; only who initiates.

**This Spark:** host `spark-1c1c`, user `guest`, wired `10.245.229.8`,
wireless `10.253.26.155`. sshd is listening on port 22.

Run everything below **on the H200**:

```bash
SPARK=guest@10.245.229.8                 # try the wired address first
SRC=/home/guest/projects/esm_proae
DEST=/scratch/$USER/esm_proae            # needs ~40 GB free
```

## B0. Check reachability before moving 31 GB

```bash
ssh $SPARK 'echo reachable; hostname'
```

Expect `reachable` and `spark-1c1c`. If it hangs or refuses, see B4 — the
addresses above are private, so they only route if both machines sit on the
same network.

## B1. The bundle

```bash
mkdir -p $DEST/data/phase1_dataset
scp -r $SPARK:$SRC/h200_bundle/* $DEST/
```

## B2. The data

Two separate commands, so the optional one can be deferred and a failure is
unambiguous.

```bash
rsync -avP --partial $SPARK:$SRC/data/phase1_dataset/dataset_100k.h5 \
  $DEST/data/phase1_dataset/          # 30.5 GB, REQUIRED

rsync -avP --partial $SPARK:$SRC/data/phase1_dataset/backbone_100k.h5 \
  $DEST/data/phase1_dataset/          # 0.7 GB, optional for now
```

Re-run either command after an interruption and it resumes.

## B3. Verify — identical to direction A

```bash
cd $DEST && md5sum -c checksums.md5
ssh $SPARK 'md5sum /home/guest/projects/esm_proae/data/phase1_dataset/dataset_100k.h5'
md5sum $DEST/data/phase1_dataset/dataset_100k.h5

export ESM_PROAE_ROOT=$DEST
export FOLDSEEK_BIN=$(which foldseek)
cd $ESM_PROAE_ROOT/code
python gate6_score_checkpoint.py --ckpt smallscale_final_40k.pt --n 100 --bs 10
```

Expect TM ~0.42-0.44, TM>0.5 ~0.32, RMSD ~11.5 A, coverage 100/100.

## B4. If the cluster cannot reach the Spark

Common: the Spark sits on a private network the cluster cannot route to, while
the Spark can reach the cluster. Open a reverse tunnel **from the Spark**:

```bash
# on the Spark, leave running
ssh -N -R 2222:localhost:22 $USER@h200.example.edu
```

Then **on the H200**, pull through it:

```bash
scp -P 2222 -r guest@localhost:$SRC/h200_bundle/* $DEST/
rsync -avP --partial -e 'ssh -p 2222' \
  guest@localhost:$SRC/data/phase1_dataset/dataset_100k.h5 \
  $DEST/data/phase1_dataset/
```

Note `scp` takes `-P` for the port while `ssh` and `rsync -e ssh` take `-p`.
A 31 GB transfer through a tunnel is slower; `--partial` still resumes, but
keep the tunnel's session alive (`tmux`, or `-o ServerAliveInterval=30`).
