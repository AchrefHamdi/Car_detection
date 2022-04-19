from turtle import width
import cv2

# importation de viseo
cap = cv2.VideoCapture("C:\\Users\\hachr\\Desktop\\projet\\car.MP4")

object_detctor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


while True:
    
    ret, frame = cap.read() # lecteur de video
    height,width,_ =frame.shape #
    print(height, width) 
    rol= frame[100:500,350:600] # les dimensions de frame à travailler

    mask = object_detctor.apply(frame) # application de mask
    _, mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY) # améliorer le mask "eliminer les shadow en gris"

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours des objets
    detections=[] # tableau des objets detecter
    # parcours les contours
    for cnt in contours:
        # supprimer petit elements
        area = cv2.contourArea(cnt)
        if (area >120 ):
         #cv2.drawContours(frame,[cnt], -1,(0,255,0),2)
         x,y,w,h = cv2.boundingRect(cnt)
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3) # mettre un rectangle cert sur les véhicule
         detections.append([x,y,w,h]) # lors de la detection ajout la position au tableau detection[]

         print(detections)

        
        #cv2.putText(rol,str("a"),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)     

    cv2.imshow("rol",rol)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)


    key=cv2.waitKey(30)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()    