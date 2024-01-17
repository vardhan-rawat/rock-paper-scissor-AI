from keras.models import load_model  
import cv2  
import numpy as np
import random
import time

#image reading and resizing 
#if shows error regarding image not found use abosolute paths
rock_img=cv2.resize(cv2.imread("rock.png"),(324,324))
paper_img=cv2.resize(cv2.imread("paper.png"),(324,324))
scissors_img=cv2.resize(cv2.imread("scissors.png"),(324,324))

#function to convert text based choice to image
def choice_img(comp_choice):
    if comp_choice=='scissors':
        return scissors_img
    if comp_choice=='rock':
        return rock_img
    if comp_choice=='paper':
        return scissors_img


#function to simulate rock paper scissors game
def rps(choice):
    choices = ['rock', 'paper', 'scissors']
    comp_choice = random.choice(choices)
    img_choice=choice_img(comp_choice)
    if choice not in choices:
        return 0,img_choice #NONE
    
    if choice == comp_choice:
        return 1,img_choice #DRAW
    
    elif (choice == 'rock' and comp_choice == 'scissors') or \
         (choice == 'paper' and comp_choice == 'rock') or \
         (choice == 'scissors' and comp_choice == 'paper'):
        return 2,img_choice #WIN
    
    else:
        return 3,img_choice #LOSE



np.set_printoptions(suppress=True)

model = load_model(r"C:\Users\vardh\Desktop\rock_paper_scissor_AI/keras_Model.h5", compile=False)
class_names = open(r"C:\Users\vardh\Desktop\rock_paper_scissor_AI/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)
#setting game variables
score=0
t_flag=0
TIMER=5

while True:
    ret, image = camera.read()
    image2=cv2.resize(image,(1000,500))

    cv2.rectangle(image2, (100,100),(424,424),color=(255,0,0), thickness=2)
    cv2.rectangle(image2, (600,100),(924,424),color=(255,0,0), thickness=2)
 
    #makeing region of intrest
    roi=image2[100:424,100:424]
    roi_resize=cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
    
    
    roi_resize = np.asarray(roi_resize, dtype=np.float32).reshape(1, 224, 224, 3)
    roi_resize = (roi_resize / 127.5) - 1

    # model for prediction

    prediction = model.predict(roi_resize)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score_perc=int(np.round(confidence_score * 100))
    choice=class_name[2:-1]

    #adjustments to code according to the models prediction accuracy

    if (choice == 'scissors') and (confidence_score_perc>=70):
        final_choice=choice       
        print(class_name[2:],end=" ")
        print(confidence_score_perc)

    elif (choice == 'rock') and (confidence_score_perc>=95):
        final_choice=choice
        print(class_name[2:],end=" ")
        print(confidence_score_perc)

    elif (choice == 'paper') and (confidence_score_perc==100):
        final_choice=choice
        print(class_name[2:],end=" ")
        print(confidence_score_perc)
    
    else:
        final_choice="None"


    #text to display in window
    text = f'Score:{score}'
    cv2.putText(image2, text, (int(image2.shape[1] / 2) + 80, image2.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    text_choice = f'Current Choice: {final_choice}'
    cv2.putText(image2, text_choice, (int(image2.shape[1] / 2) - 120, image2.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    

    #game parameters(ie. timer)
    game_input=cv2.waitKey(1)
    if game_input==115 or t_flag==1:    #press 's' to the start game!
        countdown =f'make your choice: {TIMER}'
        cv2.putText(image2, countdown, (int(image2.shape[1] / 2) - 70, image2.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if t_flag==0:
            t_flag=1
            prev=time.time()
        else:
            cur = time.time() 

            if cur-prev >= 1: 
                prev = cur 
                TIMER = TIMER-1
    #display of results
    if TIMER==0:
        result=rps(final_choice)
        comp_emo_choice=result[1]
        
        if result[0]==2:
            
            image2[100:424, 600:924] = result[1]   
            print("YOU WIN")
            cv2.putText(image2, "YOU WIN!", (int(image2.shape[1] / 2) - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            score+=1
            
        elif result[0]==3:
           image2[100:424, 600:924] = result[1]             
           cv2.putText(image2, "YOU LOST!", (int(image2.shape[1] / 2) - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else:
            image2[100:424, 600:924] = result[1] 
            cv2.putText(image2, "DRAW!", (int(image2.shape[1] / 2) - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)    

        TIMER=5
        t_flag=0
        cv2.imshow("Webcam Image", image2)
        cv2.waitKey(2000)
        


    cv2.imshow("Webcam Image", image2)
    
    keyboard_input = cv2.waitKey(2)
    
    if keyboard_input == 27:   # press 'esc' key to quit the game!
        break

camera.release()
cv2.destroyAllWindows()