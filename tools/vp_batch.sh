#!/bin/zsh
# usage: tools/vp_batch.sh <queue.txt> [--no-build] [--wait-minutes N]  (build, install, wait for unlock, run, copy back)
set -u
DEV=${VP_DEVICE:-00008112-001C38E23CC1A01E}
APP=com.michaeltesch.mtgemmvp
ROOT=${0:A:h:h}
Q=${1:?queue file}
BUILD=1
WAIT=720
shift
while (( $# )); do
  case $1 in
    --no-build) BUILD=0 ;;
    --wait-minutes) WAIT=$2; shift ;;
  esac
  shift
done
ID=$(sed -n 's/^#batch //p' $Q | head -1)
[[ -n $ID ]] || { echo "queue needs a '#batch <id>' line"; exit 2 }
OUT=$ROOT/build/vp_results/out_$ID
mkdir -p $ROOT/build/vp_results
cd $ROOT
if (( BUILD )); then
  (cd visionos && xcodegen generate -q) || exit 1
  xcodebuild -project visionos/MTGemmVP.xcodeproj -scheme MTGemmVP -destination "id=$DEV" -configuration Release \
    -derivedDataPath build/vp-dd -allowProvisioningUpdates build >build/vp_build.log 2>&1 ||
    { grep -E "error:" build/vp_build.log; echo "build failed, see build/vp_build.log"; exit 1 }
  xcrun devicectl device install app --device $DEV build/vp-dd/Build/Products/Release-xros/MTGemmVP.app >/dev/null || exit 1
fi
xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source $Q --destination Documents/queue.txt >/dev/null || exit 1
unlocked() { xcrun devicectl device info lockState --device $DEV 2>/dev/null | grep -q "passcodeRequired: false" }
t0=$(date +%s)
until unlocked; do
  (( $(date +%s) - t0 > WAIT * 60 )) && { echo "still locked after $WAIT min"; exit 3 }
  sleep 30
done
echo "unlocked at $(date)"
for attempt in 1 2 3 4; do
  xcrun devicectl device process launch --device $DEV --console --terminate-existing $APP 2>&1 | tee -a $ROOT/build/vp_results/console_$ID.txt | grep -E "^\[|rc=|batch done"
  rm -rf $OUT
  xcrun devicectl device copy from --device $DEV --domain-type appDataContainer --domain-identifier $APP \
    --source Documents/out_$ID --destination $OUT >/dev/null 2>&1
  grep -q batch-complete $OUT/state.txt 2>/dev/null && break
  echo "batch incomplete after attempt $attempt; relaunching"
  until unlocked; do sleep 30; done
done
xcrun devicectl device copy from --device $DEV --domain-type appDataContainer --domain-identifier $APP \
  --source Documents/thermal.txt --destination $OUT/thermal.txt >/dev/null 2>&1
echo "results in $OUT"
