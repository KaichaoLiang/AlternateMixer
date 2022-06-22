import json
import argparse
import string
import matplotlib.pyplot as plt

def ReadLogs(filename:string):
    mAP = []
    mAP50 = []
    mAP75 = []
    mAP_s = []
    mAP_m = []
    mAP_l=[]
    epoches = []
    with open(filename,'r') as f:
        while(True):
            logline = f.readline()
            #print(logline)
            if(not logline):
                break
            logdict = json.loads(logline)
            #print(logdict)
            if(logdict['mode'] == 'val'):
                mAP.append(logdict['0_bbox_mAP'])
                mAP50.append(logdict['0_bbox_mAP_50'])
                mAP75.append(logdict['0_bbox_mAP_75'])
                mAP_s.append(logdict['0_bbox_mAP_s'])
                mAP_m.append(logdict['0_bbox_mAP_m'])
                mAP_l.append(logdict['0_bbox_mAP_l'])
                epoches.append(logdict['epoch'])
    logData = dict()
    logData['epoches'] = epoches
    logData['mAP'] = mAP
    logData['mAP50'] = mAP50
    logData['mAP75'] = mAP75
    logData['mAP_s'] = mAP_s
    logData['mAP_m'] = mAP_m
    logData['mAP_l'] = mAP_l

    return logData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--log_file', default=None,help='the dir to save logs and models')
    args = parser.parse_args()
    print(args)
    filename = args.log_file
    print(filename)
    logData = ReadLogs(filename)
    plt.subplot(2,3,1)
    plt.plot(logData['epoches'], logData['mAP'])
    plt.title('mAP')
    
    plt.subplot(2,3,2)
    plt.plot(logData['epoches'], logData['mAP50'])
    plt.title('mAP50')

    plt.subplot(2,3,3)
    plt.plot(logData['epoches'], logData['mAP50'])
    plt.title('mAP75')

    plt.subplot(2,3,4)
    plt.plot(logData['epoches'], logData['mAP_s'])
    plt.title('mAP_s')
    
    plt.subplot(2,3,5)
    plt.plot(logData['epoches'], logData['mAP_m'])
    plt.title('mAP_m')

    plt.subplot(2,3,6)
    plt.plot(logData['epoches'], logData['mAP_l'])
    plt.title('mAP_l')

    plt.show()
    print('epoch',logData['epoches'])
    print('mAP',logData['mAP'])
    print('mAP50',logData['mAP50'])
    print('mAP75',logData['mAP75'])
    print('mAP_s',logData['mAP_s'])
    print('mAP_m',logData['mAP_m'])
    print('mAP_l',logData['mAP_l'])
