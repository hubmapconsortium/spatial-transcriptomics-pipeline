#!/bin/bash
mkdir "${@: -1}"
while (( $# > 1 ))
do
	cp -r "$1/." "${@: -1}"
	shift
done
