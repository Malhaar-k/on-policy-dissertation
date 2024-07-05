#!/bin/bash
if [ ! -d $HOME/SCFiles ]; then
        mkdir $HOME/SCFiles
fi

unzip -P iagreetotheeula SC2.4.10.zip -d $HOME/SCFiles
# unzip SC2.4.10.zip -P iagreetotheeula -d /scratch/SCFiles/StarCraftII
rm -rf /scratch/SC2.4.10.zip

export SC2PATH="$HOME/SCFiles/StarCraftII"
export MAP_DIR="$SC2PATH/Maps"

if [ ! -d $MAP_DIR/SMAC_Maps ]; then
        mkdir -p $MAP_DIR/SMAC_Maps
fi

unzip SMAC_Maps.zip -d $MAP_DIR/SMAC_Maps

echo 'StarCraft II and SMAC are installed.'