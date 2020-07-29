#!/usr/bin/env bash

# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

FOLDER=$1

echo "Folder to Download: $FOLDER"
# folders content
rm -rf "$FOLDER/*"

echo "Downloading ..."
pip install gdown
cd $FOLDER
gdown --id 1-2llitrn2l6WqE6ugCHWb9qp7fMnczQN -O model.zip
# https://drive.google.com/file/d/1-2llitrn2l6WqE6ugCHWb9qp7fMnczQN/view?usp=sharing

# Unzip archive
unzip model.zip

# Delete zip file
rm -rf model.zip
cd ~

echo "Models donwload completed ..."
