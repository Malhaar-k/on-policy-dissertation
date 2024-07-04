#!/bin/bash
unzip /scratch/SC2.4.10.zip -P iagreetotheeula -d /scratch/SCFiles/StarCraftII
rm -rf /scratch/SC2.4.10.zip

export SC2PATH="/scratch/SCFiles/StarCraftII"
export MAP_DIR="$SC2PATH/Maps"

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

unzip SMAC_Maps.zip
mkdir "$MAP_DIR/SMAC_Maps"
mv *.SC2Map "$MAP_DIR/SMAC_Maps"
echo 'StarCraft II and SMAC are installed.'