echo "0 - image is clear (cover)"
echo "1 - image has hidden message (stegocontainer)"
echo "Accuracy - percentage of covers"

echo "================SrNet================"
cd steganalysis/SRNet
python3 main_test.py

echo "================XuNet================"
cd ../XuNet_Test
python3 main_test.py

echo "================YeNet================"
cd ../YeNet_Test
python3 main_test.py

cd ../..

