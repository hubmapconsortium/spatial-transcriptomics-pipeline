#!/bin/bash

mount -o remount,size=12G /dev/shm
/opt/starfishDriver.py "$@"
