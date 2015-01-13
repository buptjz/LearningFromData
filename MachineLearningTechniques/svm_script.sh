train_bin=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/svm_bin/train
predict_bin=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/svm_bin/predict
root=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/

#for C in 0.0001 0.001 0.01 0.1 1 10
for gam in 1 10 100 1000 10000    
do
    model=format.train0.g_${gam}
    $train_bin -t 2 -g $gam -c 0.1 ${root}format.train0 ${root}${model}
    $predict_bin ${root}format.test0 ${root}${model} ${root}${model}_predict
done

#predict command
#/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/format.test4 format.train4.model format.train4.predict

#train command
#-t 2 -g 100 -c 10 /Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/format.train0

