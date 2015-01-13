train_bin=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/svm_bin/train
predict_bin=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/svm_bin/predict
root=/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/

#for C in 0.0001 0.001 0.01 0.1 1 10
for num in 0 2 4 6 8    
do
    model=format.train${num}
    $train_bin -t 1 -d 2 -g 1 -r 1 -c 0.01 ${root}${model} ${root}${model}
done

#predict command
#/Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/format.test4 format.train4.model format.train4.predict

#train command
#-t 2 -g 100 -c 10 /Volumes/BigData/Developer/LearningFromData/MachineLearningTechniques/data/format.train0

