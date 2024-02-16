# Face-Similarity
Similarity Tool for determining whether faces in photos are similar to a reference directory of face photos of the subject in question.

WINDOWS NOTE: heic images are currently not comptaible, take note of the image types you are using.

Parameters:
source_directory - Photos that will be scored on face similarity
target_directory - Where the photos that meet the cutoff and the scores will be placed
reference_directory - Photos that the source will be compared to
distance_cutoff - Cutoff value for similarity, default is .45

Download and Installation Guide:

Linux/Mac:

From the command Line:
1) Enter: git clone https://github.com/lamper-mit/Face-Similarity.git
2) Navigate to the Face-Similarity Directory
3) Enter: chmod +x install.sh
4) Enter: ./install.sh

Windows:
From Command Prompt or PowerShell:
1) Enter: git https://github.com/lamper-mit/Face-Similarity.git
2) Navigate to the Face-Similarity Directory
3) Enter: install.bat

Activating the Virtual Environment:
1) Navigate to the Face-Similarity Directory
   
Linux/Mac:

2) Enter: source venv/bin/activate

Windows:

2) Enter: venv\Scripts\activate.bat

Running the tool:
1) Activate the virutal Environment

Linux/Max:

./compare.py /path/to/source_directory /path/to/target_directory /path/to/reference_directory distance_cutoff

Windows:

python compare.py \path\to\source_directory \path\to\target_directory \path\to\reference_directory distance_cutoff
