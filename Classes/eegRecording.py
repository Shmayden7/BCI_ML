import math

class eegRecording:

    def mean(data):
        sum = 0
        for i in range(0,len(data)):
            sum += data[i]
        mean = sum/len(data)
        return mean

    def var(data):
        var = 0
        mean = eegRecording.mean(data)
        for i in range(0,len(data)):
            var += (data[i]-mean)**2

        var /= len(data) - 1
        return var
        
    def stDev(data):
        stDev =math.sqrt(eegRecording.var(data))
        return stDev

    def AROC(data):
        rise = data[len(data)-1] - data[0]
        run = len(data) - 1
        slope = rise/run
        return slope

    def integral(data): #Uses simpsons method
        sum = 0
        for i in range(0,len(data)):
            sum += data[i]
        
        return sum*(len(data)-1)



    


print(eegRecording.AROC([1,2,3,4,6]))

