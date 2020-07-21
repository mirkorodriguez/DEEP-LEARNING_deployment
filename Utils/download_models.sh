# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

environment=$1
FOLDER=$2

echo "Conda Env: $environment"
echo "Folder to Download: $FOLDER"
# borrar carpetas
rm -rf "$FOLDER/*"

echo "Downloading ..."
conda activate $environment
pip install gdown
cd $FOLDER
gdown --id 1-2llitrn2l6WqE6ugCHWb9qp7fMnczQN -O model.zip
# https://drive.google.com/file/d/1-2llitrn2l6WqE6ugCHWb9qp7fMnczQN/view?usp=sharing
conda deactivate

# Descomprimit archivos
unzip model.zip
# Borrar archivo zip
rm -rf model.zip
cd ~

echo "Models donwload completed ..."
