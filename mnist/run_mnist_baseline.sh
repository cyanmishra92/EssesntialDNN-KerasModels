clear
rm -rf *.log *.h5 *.png *.tar
python mnist_baseline.py
tar -cvf mnistBaselineTrainingLog.tar *.log *.png
tar -cvf mnistBaselineModel.tar *.h5
rm -rfv *.log
rm -rfv *.h5
rm -rfv *.png
mail -s TrainingDone cyan@psu.edu <<< 'MNIST baseline Training Completed. Copying files now'
mkdir $(date +%Y%m%d_%H%M)
mv *.tar $(date +%Y%m%d_%H%M)
