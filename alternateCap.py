#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 02:39:00 2018

@author: vivekmishra
"""

import requests

class alternateCap:
    
    def downloadCap(self,id):
        
        print(id)
        url = "http://video.google.com/timedtext?lang=en&v="+id
        r = requests.get(url)
        data = r.text
        
        return data