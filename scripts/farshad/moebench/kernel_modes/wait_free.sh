#!/bin/bash
# Exit 0 once the GNR box has been stably free for PASSES consecutive checks.
# Free = no other user holding >5 GiB RSS, whole-box idle >=90%, MemAvailable >=400 GiB.
# The sleeping happens here, in a backgrounded process -- never in a foreground tool call.
PASSES=6; INTERVAL=20; MAX=120; ok=0
for i in $(seq 1 $MAX); do
  other=$(ps -eo user,rss --no-headers | awk '$1!="farshad" && $1!="root" {r[$1]+=$2} END {m=0; for(u in r) if(r[u]>m){m=r[u]; who=u} printf "%s %.0f", (who==""?"none":who), m/1048576}')
  idle=$(top -bn2 -d 1 | awk '/^%Cpu/{v=$8} END{print int(v)}')
  avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  hog_gib=$(echo $other | cut -d' ' -f2); hog_who=$(echo $other | cut -d' ' -f1)
  printf "%s check%-3d idle=%s%% avail=%sGiB biggest_other=%s(%sGiB) streak=%d\n" \
         "$(date +%H:%M:%S)" "$i" "$idle" "$avail" "$hog_who" "$hog_gib" "$ok"
  if [ "$hog_gib" -lt 5 ] && [ "$idle" -ge 90 ] && [ "$avail" -ge 400 ]; then
    ok=$((ok+1)); [ $ok -ge $PASSES ] && { echo "BOX FREE AND STABLE"; exit 0; }
  else
    ok=0
  fi
  sleep $INTERVAL
done
echo "TIMED OUT still not stably free"; exit 1
