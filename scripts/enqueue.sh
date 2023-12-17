#!/bin/bash

PID=$1
COMMAND=${@:2}

#PID_COMMAND=`cat /proc/${PID}/cmdline | xargs -0 echo`
echo "Excute Command:bash ${COMMAND} after PID:$PID finish"

while [ -d "/proc/${PID}" ]; do
    sleep 1
done

sleep 30

bash $COMMAND