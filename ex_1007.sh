for file in {四破魚\(藍圓鰺\)2.jpg,七星鱸.jpg,香魚.jpg,DSC_0007.jpg,DSC_0130.jpg,DSC_0188.jpg}
do
    pythonw segMethod.py $file 5 5 500 50 10;
    pythonw segMethod.py $file 25 25 1000 150 30;
    pythonw segMethod.py $file 45 45 1500 250 50;
done
