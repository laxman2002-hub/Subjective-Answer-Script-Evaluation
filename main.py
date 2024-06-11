from similarity_module import NLP_Predict_Score
from ML_module import ML_Predict_Score
from adjust_score import Adjust_Score
from OCR import image_to_text # take file path as input 

# print(image_to_text("answer1.png"))

maximum_marks = 10
Cosine_sililarty_lower = 0.2
Cosine_sililarty_upper = 0.7 # Th values

solution = "Photosynthesis is a vital process for sustaining life on Earth, as it produces oxygen and organic compounds that serve as food and energy sources for organisms."
answer = "Photosynthesis produces oxygen and organic compounds that serve as food and energy sources for organisms."

ml_score = ML_Predict_Score(solution , answer)
nlp_score=NLP_Predict_Score(solution, answer, maximum_marks, Cosine_sililarty_lower, Cosine_sililarty_upper)
print(nlp_score,ml_score)

score = Adjust_Score(ml_score , nlp_score*10) # becase 0<=nlp_score<=10 , but 0<=ml_score<=100)
score = score/10 # setting score range between 0 to 10 
 
floor_v = int(score)
fr = score - floor_v
if fr>=0.75:
    fr = 1
elif fr>=0.25:
    fr = 0.5
else:
    fr = 0

final_score = floor_v + fr

print("Your Score is  = ",final_score)


