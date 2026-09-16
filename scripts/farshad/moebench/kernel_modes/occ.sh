echo "--- $(date +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg)"
top -bn1 -w 200 | awk 'NR>7 && $9>20 {printf "  %s %s %s%% cpu\n",$2,$12,$9}' | head -8
awk '/MemAvailable/{printf "  MemAvailable=%.0f GiB\n",$2/1048576}' /proc/meminfo
ps -eLo pcpu,psr --no-headers | awk '$1>3 {c[$2]+=$1} END {for(p in c) if (p+0>=1 && p+0<=95) printf "  busy cpu%s=%.0f%%\n",p,c[p]}' | sort
