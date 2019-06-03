# -*- coding: utf-8 -*-
"""

このファイルは、

pre_ud == "buy" & trade[-1] == "sell" ===> 買う


Created on Tue Apr  9 00:27:26 2019

@author: komoo
"""
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
# import pandas_talib as pt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
# from technical_indicators import technical_indicators
from DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector



trade = []
bs = []
# bs = 143.976 - 0.01

def job(x,pred,op,symbol="GBPUSD",pips=1000,loscut=10,spread=0.01,tp=30,sl=70,lot_size=0.5,use_pred=False):
    ##########################################################################    
    _zmq = DWX_ZeroMQ_Connector()
    
#     pred = pred[-300::]
    length = len(pred)
    x = x[-length::,]
    
    
    
    ud = []
    pre_ud = []
    pre_ud2 =  []
    
    for i in range(1,len(pred)):
        ud.extend(np.where(x[i]>x[i-1],"buy","sell"))
        pre_ud.extend(np.where(pred[i]<x[i-1],"buy","sell"))
        pre_ud2.extend(np.where(pred[i]>pred[i-1],"buy","sell"))

    
    print("accuracy = ",sklearn.metrics.accuracy_score(ud,pre_ud))
    pre_ud2 = pre_ud
    x = op[-len(pre_ud)::]
    
    ##########################################################################
    ##########################################################################
    length = len(trade)

    if length == 0:
        if pre_ud[-1] == "buy" and  pre_ud2[-1] == "buy":
            # _type = 0 <== "buy"
            _zmq._DWX_MTX_SEND_COMMAND_(_symbol=symbol,_lots=lot_size,_SL=sl,_TP=tp)
            bs.append(np.float(x[-1]) - spread)
            trade.append("buy")
            
        elif pre_ud[-1] == "sell" and pre_ud2[-1] == "sell":
            # _type = 1 <== "sell"
            _zmq._DWX_MTX_SEND_COMMAND_(_symbol=symbol,_lots=lot_size,_type=1,_SL=sl,_TP=tp)
            bs.append(np.float(x[-1]) + spread)
            trade.append("sell")
    
    else:
        if pre_ud[-1] == "buy" and  pre_ud2[-1] == "buy" and trade[-1] == "buy":
            pip = (np.float(x[-1]) - bs[-1])*pips
            if pip <=  -loscut:
                _zmq._DWX_MTX_CLOSE_ALL_TRADES_()
            trade.append("buy")                
        elif pre_ud[-1] == "buy" and  pre_ud2[-1] == "buy" and trade[-1] == "sell":
            _zmq._DWX_MTX_CLOSE_ALL_TRADES_()
            _zmq._DWX_MTX_SEND_COMMAND_(_symbol=symbol,_lots=lot_size,_SL=sl,_TP=tp)
            bs.append(np.float(x[-1]) - spread)
            trade.append("buy")

        elif pre_ud[-1] == "sell" and pre_ud2[-1] == "sell" and trade[-1] == "sell":
            pip = (bs[-1] - np.float(x[-1]))*pips
            if  pip <= -loscut:
                _zmq._DWX_MTX_CLOSE_ALL_TRADES_()
            print("pip = ",pip)
            trade.append("sell")
                        
        elif pre_ud[-1] == "sell" and pre_ud2[-1] == "sell" and trade[-1] == "buy":
            _zmq._DWX_MTX_CLOSE_ALL_TRADES_()
            _zmq._DWX_MTX_SEND_COMMAND_(_symbol=symbol,_lots=lot_size,_type=1,_SL=sl,_TP=tp)
            bs.append(np.float(x[-1]) + spread)
            trade.append("sell")
            

    print("trade length = ",len(trade))
    return trade,bs
    
#     _zmq._DWX_MTX_SEND_COMMAND_(_symbol=symbol,_lots=lot_size,_SL=sl,_TP=100000)

##############################################################################
#                   use trade = job(trade=trade)                             #
##############################################################################