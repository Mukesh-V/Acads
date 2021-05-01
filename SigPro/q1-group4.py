from scipy.io.wavfile import read as read_wav
import os
import numpy as np

from dft_quiz import DiscreteFourierTransformQuiz
sr, data = read_wav('sargam.wav')
print(sr)

props = { 'N' : sr }
obj = DiscreteFourierTransformQuiz(props)

obj.setDataPoints(data[sr*6: sr*7+1])
obj.setNFouriers()
obj.plotFourier()

obj.setDataPoints(data[sr*7: sr*8+1])
obj.setNFouriers()
obj.plotFourier()