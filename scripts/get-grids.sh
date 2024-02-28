#!/bin/bash
mir=/usr/local/apps/mars/versions/6.33.17.6/bin/mir

if [[ ! -f 2t.grib ]]; then
mars<<EOF
retrieve,levtype=sfc,param=2t,target=tmp.grib
EOF
mv tmp.grib 2t.grib
fi

for n in o16 o32 o48 o96
do
    echo "Processing $n"
    if [[ ! -f $n.grib ]]; then
        $mir --grid=$n 2t.grib tmp.grib
        mv tmp.grib $n.grib
    fi
    if [[ ! -f $n.npz ]]; then
    ./get-grids.py $n
    fi
    echo "Done $n"
done
