from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):

    name : str ="Subbu"#default value
    age : Optional[int] = None
    email: EmailStr
    cgpa : float = Field(gt=0 ,lt=10)

new_student={'age':'22','email':'subbu@gmail.com','cgpa':8}#pass value 

student=Student(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json()
print(student_dict['age'])
print(student_json)
