import time
import datetime
from collections import OrderedDict

import numpy as np

def include_element(element):
    if element.find('-----------') >= 0:
        return False
    elif element.find('Log') >= 0:
        return False
    else:
        return True

def define_columns_and_ids(numCams, legacy=False):
    # cols = ['TS', ['measx_mm','scorex', 'measy_mm','scorey','cmeasx_mm','cscorex','cmeasy_mm','cscorey', 'x unwrap cnt', 'y unwrap cnt', 'post measx_mm', 'post measy_mm', 'post cmeasx_mm', 'post cmeasy_mm']]
    # col_ids = {}
    # for idx, col in enumerate(cols[1]):
    #     col_ids[col] = idx
    # print(col_ids)
    
    cols = ['TS', 'time_s']

    if not legacy:
        cols.extend(['time_offset_s', 'numread', 'TTMNumComplete', 'TTMNumPartial'])    

    cam_fields = ['measx_mm','scorex', 'measy_mm','scorey','cmeasx_mm','cscorex','cmeasy_mm','cscorey', 'x unwrap cnt', 'y unwrap cnt', 'post measx_mm', 'post measy_mm', 'post cmeasx_mm', 'post cmeasy_mm']

    for i in range(numCams):
        for cam_field in cam_fields:
            cols.append("C%d %s" % (i, cam_field))
    # print(cols)

    col_ids = OrderedDict()
    for idx, col in enumerate(cols):
        col_ids[col] = idx

    return cols, col_ids

def parse_into_data_sets(data):
    data_ranges = []
    max_delta = 1
    start = 0
    found_first = False
    for idx, d in enumerate(data):
        if len(d) <= 1:
            print("Found blank line: ", d)
            continue

        if found_first == False:
            found_first = True
            d_prev = d
        else:
            try:
                ts1 = d_prev.split(',')[0][-23:]
                ts2 = d.split(',')[0][-23:]
                dt1 = datetime.datetime.strptime(ts1, "%Y-%m-%d %H:%M:%S.%f")
                dt2 = datetime.datetime.strptime(ts2, "%Y-%m-%d %H:%M:%S.%f")
                s1 =time.mktime(dt1.timetuple()) + (dt1.microsecond / 1000000.0)
                s2 =time.mktime(dt2.timetuple()) + (dt2.microsecond / 1000000.0)
                #delta = float(d.split(',')[0][-6:]) - float(d_prev.split(',')[0][-6:])
                delta = s2 - s1
                if delta > max_delta:
                    end = idx - 1
                    data_ranges.append((start, end))
                    start = idx
                d_prev = d
            except ValueError:
                print("ValueError!!: %s" % d)
    data_ranges.append((start, idx))
    len(data_ranges)

    data_sets = []
    for idx, data_range in enumerate(data_ranges):
        data_sets.append(data[data_range[0]:data_range[1]])
        print("Data set %d, range: %d - %d (total: %d)" % (idx, data_range[0], data_range[1], (data_range[1]-data_range[0])))
        
    for idx, ds in enumerate(data_sets):
        print("[%d] range: %s -> %s" % ( idx, ds[0].split(',')[0], ds[-1].split(',')[0]))
        
    return data_ranges, data_sets

def cleanup_and_format_data_set(data_set, legacy=False):
    data_set_formated = []
    for idx, data_row_str in enumerate(data_set):
        if data_row_str.find('-----------') >= 0 or data_row_str.find('Log Starting') >= 0:
            continue

        data_row_formatted = []
        data_row = data_row_str.split(',')
        #print(idx)
        data_row_formatted.append(data_row[0][17:])
        
        if idx == 0:
            dt0 = datetime.datetime.strptime(data_row[0][17:], "%Y-%m-%d %H:%M:%S.%f")
            t0 =time.mktime(dt0.timetuple()) + (dt0.microsecond / 1000000.0)
            data_row_formatted.append(0)
        else:
            dt1 = datetime.datetime.strptime(data_row[0][17:], "%Y-%m-%d %H:%M:%S.%f")
            t1 = time.mktime(dt1.timetuple()) + (dt1.microsecond / 1000000.0)
            data_row_formatted.append(t1-t0)

        if legacy:
            offset = 0
        else:
            data_row_formatted.append(float(data_row[2]))                 
            data_row_formatted.append(int(data_row[3]))                 
            data_row_formatted.append(int(data_row[4]))                 
            data_row_formatted.append(int(data_row[5]))                 
            offset = 4
        
        data_row_formatted.append(float(data_row[offset+2]))                 
        data_row_formatted.append(int(data_row[offset+3]))                   
        data_row_formatted.append(float(data_row[offset+4]))                 
        data_row_formatted.append(int(data_row[offset+5]))                   
        data_row_formatted.append(float(data_row[offset+6]))                 
        data_row_formatted.append(int(data_row[offset+7]))
        data_row_formatted.append(float(data_row[offset+8]))                 
        data_row_formatted.append(int(data_row[offset+9]))                   
        data_row_formatted.append((data_row[offset+10].count('0') - 1) * -1)
        data_row_formatted.append((data_row[offset+11].count('0') - 1) * -1)
        data_row_formatted.append(float(data_row[offset+13]))                
        data_row_formatted.append(float(data_row[offset+14]))                
        data_row_formatted.append(float(data_row[offset+15]))                
        data_row_formatted.append(float(data_row[offset+16]))                
        data_row_formatted.append(float(data_row[offset+18]))                
        data_row_formatted.append(int(data_row[offset+19]))                  
        data_row_formatted.append(float(data_row[offset+20]))                
        data_row_formatted.append(int(data_row[offset+21]))                  
        data_row_formatted.append(float(data_row[offset+22]))                
        data_row_formatted.append(int(data_row[offset+23]))                  
        data_row_formatted.append(float(data_row[offset+24]))                
        data_row_formatted.append(int(data_row[offset+25]))                  
        data_row_formatted.append((data_row[offset+26].count('0') - 1) * -1)
        data_row_formatted.append((data_row[offset+27].count('0') - 1) * -1)
        data_row_formatted.append(float(data_row[offset+29]))                
        data_row_formatted.append(float(data_row[offset+30]))                
        data_row_formatted.append(float(data_row[offset+31]))                
        data_row_formatted.append(float(data_row[offset+32]))                
        data_row_formatted.append(float(data_row[offset+34]))                
        data_row_formatted.append(int(data_row[offset+35]))                  
        data_row_formatted.append(float(data_row[offset+36]))                
        data_row_formatted.append(int(data_row[offset+37]))                  
        data_row_formatted.append(float(data_row[offset+38]))                
        data_row_formatted.append(int(data_row[offset+39]))                  
        data_row_formatted.append(float(data_row[offset+40]))                
        data_row_formatted.append(int(data_row[offset+41]))                  
        data_row_formatted.append((data_row[offset+42].count('0') - 1) * -1)
        data_row_formatted.append((data_row[offset+43].count('0') - 1) * -1)
        data_row_formatted.append(float(data_row[offset+45]))                
        data_row_formatted.append(float(data_row[offset+46]))                
        data_row_formatted.append(float(data_row[offset+47]))                
        data_row_formatted.append(float(data_row[offset+48]))                
        data_row_formatted.append(float(data_row[offset+50]))                
        data_row_formatted.append(int(data_row[offset+51]))                  
        data_row_formatted.append(float(data_row[offset+52]))                
        data_row_formatted.append(int(data_row[offset+53]))                  
        data_row_formatted.append(float(data_row[offset+54]))                
        data_row_formatted.append(int(data_row[offset+55]))                  
        data_row_formatted.append(float(data_row[offset+56]))                
        data_row_formatted.append(int(data_row[offset+57]))                  
        data_row_formatted.append((data_row[offset+58].count('0') - 1) * -1)
        data_row_formatted.append((data_row[offset+59].count('0') - 1) * -1)
        data_row_formatted.append(float(data_row[offset+61]))                
        data_row_formatted.append(float(data_row[offset+62]))                
        data_row_formatted.append(float(data_row[offset+63]))                
        data_row_formatted.append(float(data_row[offset+64]))
        data_set_formated.append(data_row_formatted)
    return data_set_formated
        
def cleanup_and_format_data_row_formatted(data_set):
    data_set_formated = []
    for idx, data_row_str in enumerate(data_set):
        data_row_formatted = []
        data_row = data_row_str.split(',')
        #print(idx)
        
        data_row_formatted.append(data_row[0][17:])

        cam1_data = []
        cam1_data.append(float(data_row[2]))                 
        cam1_data.append(int(data_row[3]))                   
        cam1_data.append(float(data_row[4]))                 
        cam1_data.append(int(data_row[5]))                   
        cam1_data.append(float(data_row[6]))                 
        cam1_data.append(int(data_row[7]))
        cam1_data.append(float(data_row[8]))                 
        cam1_data.append(int(data_row[9]))                   
        cam1_data.append((data_row[10].count('0') - 1) * -1)
        cam1_data.append((data_row[11].count('0') - 1) * -1)
        cam1_data.append(float(data_row[13]))                
        cam1_data.append(float(data_row[14]))                
        cam1_data.append(float(data_row[15]))                
        cam1_data.append(float(data_row[16]))
        data_row_formatted.append(cam1_data)
        
        cam2_data = []
        cam2_data.append(float(data_row[18]))                
        cam2_data.append(int(data_row[19]))   
        cam2_data.append(float(data_row[20]))                
        cam2_data.append(int(data_row[21]))                  
        cam2_data.append(float(data_row[22]))                
        cam2_data.append(int(data_row[23]))                  
        cam2_data.append(float(data_row[24]))                
        cam2_data.append(int(data_row[25]))                  
        cam2_data.append((data_row[26].count('0') - 1) * -1)
        cam2_data.append((data_row[27].count('0') - 1) * -1)
        cam2_data.append(float(data_row[29]))                
        cam2_data.append(float(data_row[30]))                
        cam2_data.append(float(data_row[31]))                
        cam2_data.append(float(data_row[32]))
        data_row_formatted.append(cam2_data)
        
        cam3_data = []
        cam3_data.append(float(data_row[34]))                
        cam3_data.append(int(data_row[35]))                  
        cam3_data.append(float(data_row[36]))                
        cam3_data.append(int(data_row[37]))                  
        cam3_data.append(float(data_row[38]))                
        cam3_data.append(int(data_row[39]))                  
        cam3_data.append(float(data_row[40]))                
        cam3_data.append(int(data_row[41]))                  
        cam3_data.append((data_row[42].count('0') - 1) * -1)
        cam3_data.append((data_row[43].count('0') - 1) * -1)
        cam3_data.append(float(data_row[45]))                
        cam3_data.append(float(data_row[46]))                
        cam3_data.append(float(data_row[47]))                
        cam3_data.append(float(data_row[48]))
        data_row_formatted.append(cam3_data)
        
        cam4_data = []
        cam4_data.append(float(data_row[50]))                
        cam4_data.append(int(data_row[51]))                  
        cam4_data.append(float(data_row[52]))                
        cam4_data.append(int(data_row[53]))                  
        cam4_data.append(float(data_row[54]))                
        cam4_data.append(int(data_row[55]))                  
        cam4_data.append(float(data_row[56]))                
        cam4_data.append(int(data_row[57]))                  
        cam4_data.append((data_row[58].count('0') - 1) * -1)
        cam4_data.append((data_row[59].count('0') - 1) * -1)
        cam4_data.append(float(data_row[61]))                
        cam4_data.append(float(data_row[62]))                
        cam4_data.append(float(data_row[63]))                
        cam4_data.append(float(data_row[64]))
        data_row_formatted.append(cam4_data)
        
        data_set_formated.append(data_row_formatted)
    return data_set_formated

def validate_columns_rough(data_sets_rough): 
    c = {}
    for idx, ds in enumerate(data_sets_rough):
        for d in ds:
            cnt = d.count(',')
            if cnt not in c:
                c[cnt] = 1
            else:
                c[cnt] += 1
    #print(c)
    return c
    
def validate_columns_cleanedup(data_sets):
    c = {}
    for idx, ds in enumerate(data_sets):
        for d in ds:
            #cnt = d.count(',')
            cnt = len(d)
            if cnt not in c:
                c[cnt] = 1
            else:
                c[cnt] += 1
    #print(c)
    return c

TTM_MoirePeriod_mm = 0.001
TTM_CoarseOffset_mm = 0

def unwrap(meas_mm, cmeas_mm, cscore):
    cnt = 0
    if cscore > 0:
        meas_mm = -np.nan
    else:
        err = cmeas_mm + TTM_CoarseOffset_mm - meas_mm
        while (np.fabs(err) > 0.5*TTM_MoirePeriod_mm) and (cnt < 4):
            if err > 0.0:
                meas_mm += TTM_MoirePeriod_mm
            else:
                meas_mm -= TTM_MoirePeriod_mm

            err = cmeas_mm + TTM_CoarseOffset_mm - meas_mm
            cnt += 1
    return cnt, meas_mm