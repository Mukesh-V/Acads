from scipy.io.wavfile import read as read_wav
import os
import numpy as np

from dft_quiz import DiscreteFourierTransformQuiz
sr, data = read_wav('sargam.wav')
print(sr)

props = { 'N' : sr }
obj = DiscreteFourierTransformQuiz(props)

obj.setDataPoints(data[sr*2: sr*3+1])
obj.setNFouriers()
obj.plotFourier()

obj.setDataPoints(data[sr*3: sr*4+1])
obj.setNFouriers()
obj.plotFourier()