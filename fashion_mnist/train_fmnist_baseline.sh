clear
rm -rf *.log *.h5 *.png *.tar
python fmnist_baseline.py
tar -cvf fmnistBaselinTrainingLog.tar *.log *.png
tar -cvf fmnistBaselineModel.tar *.h5
rm -rfv *.log
rm -rfv *.h5
rm -rfv *.png
mail -s TrainingDone cyan@psu.edu <<< 'fashion_MNIST baseline Training Completed. Copying files now'
mkdir $(date +%Y%m%d_%H%M)
mv *.tar $(date +%Y%m%d_%H%M)
