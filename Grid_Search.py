import subprocess
from itertools import product
import os
import shutil
import cv2
def save_detections(pos,neg,cascade_path):  
        minneigh=3
        min_x=min_y=15
        cascade = cv2.CascadeClassifier(cascade_path)      
        for i in range(minneigh,minneigh+6):
                path=os.path.join('/home/gabrbrr/RoboCup/Leg_sim_2/HAAR_detections','minneigh_'+str(i)+str(test(pos,neg,cascade_path,i)))
                if not os.path.exists(path):       
                        os.makedirs(path)
                for image_filename in os.listdir(pos):
                        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                                image_path = os.path.join(pos, image_filename)
                                img = cv2.imread(image_path)       
                                lbp_features = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=i, minSize=(min_x, min_y))


                                for (x, y, w, h) in lbp_features:
                                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                                print(os.path.join(path,image_filename))
                                cv2.imwrite(os.path.join(path,image_filename),img)
                                
                                

                for image_filename in os.listdir(neg):
                        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                                image_path = os.path.join(neg, image_filename)
                                img = cv2.imread(image_path)        
                                lbp_features = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=i, minSize=(min_x, min_y))


                                for (x, y, w, h) in lbp_features:
                                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                                           
                                cv2.imwrite(os.path.join(path,image_filename),img)

def delete_xml(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)

def copy_to(source_file_path,destination_directory,new_file_name):

    destination_file_path = os.path.join(destination_directory, new_file_name)

    shutil.copy(source_file_path, destination_file_path)

def test(pos,neg,cascade_path,minNeigh):  
        cascade = cv2.CascadeClassifier(cascade_path)      
        fn=tn=tp=fp=0
        for image_filename in os.listdir(pos):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                        image_path = os.path.join(pos, image_filename)
                        img = cv2.imread(image_path)
                        objects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=minNeigh, minSize=(10,5))
                        if len(objects)==0:
                                fn=fn+1
                        else:
                                tp=tp+1
        for image_filename in os.listdir(neg):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                        image_path = os.path.join(neg, image_filename)
                        img = cv2.imread(image_path)
                        objects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=minNeigh, minSize=(10,5))
                        if len(objects)==0:
                                tn=tn+1
                        else:
                                fp=fp+1
        epsilon=0.00001
        return ((tp+tn)/(epsilon+tp+tn+fp+fn),tp/(epsilon+tp+fp) ,tp/(epsilon+tp+fn))

def show_detections(pos,neg,cascade_path):  
        cascade = cv2.CascadeClassifier(cascade_path)      
        for image_filename in os.listdir(pos):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                        image_path = os.path.join(pos, image_filename)
                        img = cv2.imread(image_path)       
                        lbp_features = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 5))


                        for (x, y, w, h) in lbp_features:
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                        cv2.imshow("Detected LBP Features", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

        for image_filename in os.listdir(neg):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                        image_path = os.path.join(neg, image_filename)
                        img = cv2.imread(image_path)        
                        lbp_features = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 5))


                        for (x, y, w, h) in lbp_features:
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                        cv2.imshow("Detected LBP Features", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

test_pos="/home/gabrbrr/RoboCup/Leg_sim_2/Test_Positives"
test_neg="/home/gabrbrr/RoboCup/Leg_sim_2/Test_Negatives"
target_dir="/home/gabrbrr/RoboCup/Leg_sim_2"
out_path=os.path.join(target_dir,"output_directory")
cascade_file_path=os.path.join(out_path,"cascade.xml")
delete_xml(out_path)
min_num_stages=2
max_num_stages=30
feature_type=['HAAR,LBP']
bt = [ 'DAB','RAB','GAB','LB']
num_pos = [164]  
num_neg= [300]  
max_depth=[1,2,3]
maxFalseAlarmRate=[0.3,0.35,0.4]
max_weak_count = [20,70,100]
min_num_minNeighbours=2
max_num_minNeighbours=15
w=30
h=30
max_acc_LBP=max_precision_LBP=max_recall_LBP=0
max_acc_HAAR=max_precision_HAAR=max_recall_HAAR=0
with open(os.path.join(target_dir,"grid_search_results.txt"), "w") as results_file:
    for bt, feature_type, num_pos, num_neg, max_depth, max_weak_count, maxFalseAlarmRate in product(
        bt, feature_type, num_pos, num_neg, max_depth, max_weak_count, maxFalseAlarmRate,
    ):
        local_max_acc_LBP=local_max_recall_LBP=local_max_precision_LBP=0
        local_max_acc_HAAR=local_max_recall_HAAR=local_max_precision_HAAR=0
        best_stage=min_num_stages
        for stage in range(min_num_stages,max_num_stages):
            
            command = [
                'opencv_traincascade',
                '-data', 'output_directory',
                '-vec', 'positive_samples_30_30_164.vec',
                '-bg', 'negatives.txt',
                '-bt', bt,
                '-featureType', feature_type,
                '-numPos', str(num_pos),
                '-numNeg', str(num_neg),
                '-w', str(w), 
                '-h', str(h),  
                '-maxDepth', str(max_depth),
                '-maxWeakCount', str(max_weak_count),
                '-numStages', str(stage) ,
                '-maxFalseAlarmRate', str(maxFalseAlarmRate),              
            ]

            try:
                local_max_acc_LBP_neigh=local_max_precision_LBP_neigh=local_max_recall_LBP_neigh=0
                local_max_acc_HAAR_neigh=local_max_precision_HAAR_neigh=local_max_recall_HAAR_neigh=0
                for neigh in range(min_num_minNeighbours,max_num_minNeighbours):   
                        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True,cwd=target_dir)
                        Hyperparameters=f"BT{bt}_FeatureType{feature_type}_NumPos{num_pos}_NumNeg{num_neg}_MaxDepth{max_depth}_MaxWeakCount{max_weak_count}_w{w}_h{h}_maxFalseAlarmRate{maxFalseAlarmRate}_neigh{neigh}_stage{stage}"
                        
                        accuracy,precision,recall=test(test_pos,test_neg,cascade_file_path,neigh)
                        results_file.write(f"Hyperparameters:{Hyperparameters}_accuracy{accuracy}_precision{precision}_recall{recall}\n")
                        if(feature_type=='LBP' and (accuracy,precision,recall)>(local_max_acc_LBP_neigh,local_max_precision_LBP_neigh,local_max_recall_LBP_neigh)):
                                best_neigh=neigh
                                local_max_acc_LBP_neigh=accuracy
                                local_max_recall_LBP_neigh=recall
                                local_max_precision_LBP_neigh=precision
                                if  (local_max_acc_LBP_neigh,local_max_precision_LBP_neigh,local_max_recall_LBP_neigh)>(max_acc_LBP,max_precision_LBP,max_recall_LBP):
                                        max_acc_LBP=local_max_acc_LBP_neigh
                                        max_recall_LBP=local_max_recall_LBP_neigh
                                        max_precision_LBP=local_max_precision_LBP_neigh
                                        copy_to(cascade_file_path,os.path.join(target_dir,"Best_LBP"),Hyperparameters)
                                        print("new_max LBP",Hyperparameters,max_acc_LBP,max_recall_LBP,max_precision_LBP)
                
                        elif    (feature_type=='HAAR' and (accuracy,precision,recall)>(local_max_acc_HAAR_neigh,local_max_precision_HAAR_neigh,local_max_recall_HAAR_neigh)):
                                best_neigh=neigh
                                local_max_acc_HAAR_neigh=accuracy
                                local_max_recall_HAAR_neigh=recall
                                local_max_precision_HAAR_neigh=precision
                                if (local_max_acc_HAAR_neigh,local_max_precision_HAAR_neigh,local_max_recall_HAAR_neigh)>(max_acc_HAAR,max_precision_HAAR,max_recall_HAAR):
                                        max_acc_HAAR=local_max_acc_HAAR_neigh
                                        max_recall_HAAR=local_max_recall_HAAR_neigh
                                        max_precision_HAAR=local_max_precision_HAAR_neigh
                                        copy_to(cascade_file_path,os.path.join(target_dir,"Best_HAAR"),Hyperparameters)
                                        print("new_max HAAR",Hyperparameters, max_acc_HAAR,max_recall_LBP,max_precision_HAAR)
                        
                if(feature_type=='LBP' and (local_max_acc_LBP_neigh,local_max_precision_LBP_neigh,local_max_recall_LBP_neigh)>(local_max_acc_LBP,local_max_precision_LBP,local_max_recall_LBP)):
                        local_max_acc_LBP=local_max_acc_LBP_neigh
                        local_max_recall_LBP=local_max_recall_LBP_neigh
                        local_max_precision_LBP=local_max_precision_LBP_neigh
                        
                elif(feature_type=='HAAR' and (local_max_acc_HAAR_neigh,local_max_precision_HAAR_neigh,local_max_recall_HAAR_neigh)>(local_max_acc_HAAR,local_max_precision_HAAR,local_max_recall_HAAR)):
                        local_max_acc_HAAR=local_max_acc_HAAR_neigh
                        local_max_recall_HAAR=local_max_recall_HAAR_neigh
                        local_max_precision_HAAR=local_max_precision_HAAR_neigh
                        
                else : break

                          
                    
            except subprocess.CalledProcessError as e:
                    results_file.write(f"Error occurred for hyperparameters: BT={bt}, FeatureType={feature_type}, "
                                    f"NumPos={num_pos}, NumNeg={num_neg}"
                                    f"MaxDepth={max_depth}, MaxWeakCount={max_weak_count}\n")
                    results_file.write(e.output)

            delete_xml(out_path)
        

       
        delete_xml(out_path)



