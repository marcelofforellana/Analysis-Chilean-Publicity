# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:57:11 2021

@author: marce
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 03:05:04 2021

@author: Marcelo Orellana (marceorellana@alumnos.uai.cl)

Title: Crawling Images(Lun.com)

"""

import random
from time import sleep
from selenium import webdriver
from datetime import date, timedelta
import pandas as pd
import os
import urllib
import shutil

from datetime import date, timedelta

# Get the period of webscrapping

# Start dates

initial_day= int(input("Enter the initial day: "))
initial_month= int(input("Enter initial month: "))
initial_year= int(input("Enter initial year: "))



# Final dates
final_day= int(input("Enter final day: "))
final_month= int(input("Enter final month: "))
final_year= int(input("Enter final year: "))


start_date = date(initial_year,initial_month,initial_day)
end_date = date(final_year, final_month,final_day)

delta = timedelta(days=1)

target_dates= list()

while start_date <= end_date:
  current_date=start_date.strftime("%Y-%m-%d")
  # print (current_date)
  target_dates.append(current_date)
  start_date += delta

#  1-5 each month
aux=list()
for i in target_dates:
  target_split= i.split("-")
  if int(target_split[2])in range(1,6):
    aux.append(i)
print(aux)

path_main_folder= 'D:\Scraping_Images'

driver = webdriver.Chrome(r'C:\Users\marce\Desktop\Tesis\Webscraping Images\chromedriver.exe') 

for day_i in aux:
    
    link='https://www.lun.com/Pages/NewsDetail.aspx?dt='+day_i+'&SupplementId=0&PaginaId=4'
   
    
    driver.get(link)
    
    boton = driver.find_element_by_xpath('//*[@id="topBar-rightArrow"]')

    len_news= len(driver.find_elements_by_xpath('//td[@align="left"]//div[@class="SmallImages_News"]'))+1
    
    print(len_news)
    
    os.chdir(path_main_folder)
    
    if os.path.exists(day_i):
        shutil.rmtree(day_i)
      
    os.makedirs(day_i)
    
    save_folder=day_i
    
    topic=day_i
    
    count=0
    
    for i in range(4,len_news): # clicking to load the next pages # original=2
        try:
            # Getting images
            elem1 = driver.find_element_by_id('image-gallery')
            sub = elem1.find_elements_by_tag_name('img')
            
            anunciante= driver.find_element_by_id('anunciante_av')
    
            anunciante=anunciante.get_attribute('innerHTML')
            
         
            for j,i in enumerate(sub):
                src = i.get_attribute('src')                         
                try:
                    if src != None:
                        src  = str(src)
                        print(src)
                        
                        if len(anunciante)!=0:
                            urllib.request.urlretrieve(src, os.path.join(save_folder, topic+"-"+str(count)+"-"+anunciante+'.jpg'))
                            count+=1
                        else:
                            urllib.request.urlretrieve(src, os.path.join(save_folder, topic+"-"+str(count)+'.jpg'))
                            count+=1
                            
    
                except Exception as e:              #catches type error along with other errors
                    print(f'fail with error {e}') 
            
            # le doy click
            boton.click()
            # wait to load dynamic information
            sleep(random.uniform(8.0, 10.0))
            # clicking again 
            boton = driver.find_element_by_xpath('//div[@id="topBar-rightArrow"]')
        except:
            # if there is a mistake, break the loop
            break