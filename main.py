import os
import sys
from prettytable import PrettyTable

sys.path.insert(0, 'FingerWriting/')
import FingerPersonalized
print('Finger-Writing Tests Started')
fpAvgL,fpSdL,fpAvgU,fpSdU=FingerPersonalized.main()
print('Finger-Writing Tests Finished')


x = PrettyTable()
x.field_names = ["", "Personal-Lowercase","Personal-Uppercase"]


x.add_row(["Finger-Writing","{0:.0%}".format(fpAvgL),
	"{0:.0%}".format(fpAvgU)])


print(x)