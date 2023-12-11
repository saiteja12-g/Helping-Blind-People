from gtts import gTTS
from io import BytesIO
import pygame
import time


def wait():
    while pygame.mixer.get_busy():
        time.sleep(1)


def speak(text, language='en'):
    ''' speaks without saving the audio file '''
    mp3_fo = BytesIO()
    tts = gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    mp3_fo.seek(0)
    sound = pygame.mixer.Sound(mp3_fo)
    sound.play()
    wait()

def speaksave(text, language='en'):
    ''' saves the audio file and then speaks '''
    tts = gTTS(text, lang=language)
    tts.save("myfile.mp3")
    sound = pygame.mixer.Sound("myfile.mp3")
    sound.play()
    wait()


pygame.init()
pygame.mixer.init()
# speaksave("This audio is saved as myfile.mp3, look in the folder")
# speak("This audio is not saved")