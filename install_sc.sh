#!/bin/bash

export SC2PATH="$PWD/SCFiles/StarCraftII"
export MAP_DIR="$SC2PATH/Maps"

if [ ! -d $SC2PATH ]; then
        mkdir -p $SC2PATH
fi

unzip $PWD/SC2.4.10.zip -P iagreetotheeula -d $SC2PATH



if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

unzip SMAC_Maps.zip -d $MAP_DIR
mkdir "$MAP_DIR/SMAC_Maps"
mv *.SC2Map "$MAP_DIR/SMAC_Maps"
rm -rf /scratch/SC2.4.10.zip

echo 'StarCraft II and SMAC are installed.'