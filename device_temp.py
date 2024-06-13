import os
from time import sleep

def print_and_max(printit=True):
    max_temp_celsius, max_temp_fahrenheit = 0, 0       
    for k in range(7):
        with open(f'/sys/class/thermal/thermal_zone{k}/temp') as temp_file:
            if printit: print(f'\tthermal_zone{k}')
            for line in temp_file:
                celsius = float(line)/1000
                fahrenheit = 9/5*celsius+32
                if printit:
                    print('\t\t', celsius, '℃')
                    print('\t\t', fahrenheit, '℉')
                if celsius > max_temp_celsius:
                    max_temp_celsius, max_temp_fahrenheit = celsius, fahrenheit
    return max_temp_celsius, max_temp_fahrenheit

if __name__ == "__main__":
    while True:
        print()
        print()
        celsius, fahrenheit = print_and_max()
        print("max temps:")
        print('\t', celsius, '℃')
        print('\t', fahrenheit, '℉')
        print()        
        sleep(1)
        print("\033[H\033[J", end="")
        if(celsius >= 85):
            print("\a")
            print("COMPONENT OVERHEATING!")
            print()