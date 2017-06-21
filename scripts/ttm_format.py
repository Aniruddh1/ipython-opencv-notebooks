#!/usr/bin/env /usr/bin/python

import sys
import time
import datetime
import csv
from os.path import expanduser, split

def cleanup_and_format_data_set(data_set):
    data_set_formated = []
    for idx, data_row_str in enumerate(data_set):
        data_row_formatted = []
        data_row = data_row_str.split(',')
        #print(idx)
        #data_row_formatted.append(data_row[0][17:])
        
        if idx == 0:
            dt0 = datetime.datetime.strptime(data_row[0][17:], "%Y-%m-%d %H:%M:%S.%f")
            t0 =time.mktime(dt0.timetuple()) + (dt0.microsecond / 1000000.0)
            data_row_formatted.append(float(data_row[1]))
            data_row_formatted.append(0)
            t_prev = t0
        else:
            dt1 = datetime.datetime.strptime(data_row[0][17:], "%Y-%m-%d %H:%M:%S.%f")
            t1 = time.mktime(dt1.timetuple()) + (dt1.microsecond / 1000000.0)
            data_row_formatted.append(float(data_row[1]) + (t1-t0))
            data_row_formatted.append(t1 - t_prev)
            t_prev = t1

        data_row_formatted.append(float(data_row[1]))                 
        data_row_formatted.append(int(data_row[2]))                 
        data_row_formatted.append(int(data_row[3]))                 
        data_row_formatted.append(int(data_row[4]))                 
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

def main():
    if len(sys.argv) == 1:
        print("Error! No template ID passed as argument")
        sys.exit()

    template_id = sys.argv[1]
    ttm_raw_path       = '%s/workspace/stats/template.%s/wafer.1/imprint.1/TTMLogStream.1' % (expanduser("~"), template_id)
    ttm_formatted_path = '%s/workspace/stats/template.%s/wafer.1/imprint.1/TTMLogStream.dat' % (expanduser("~"), template_id)

    print("ttm_formatted_path: ", ttm_formatted_path)

    with  open(ttm_raw_path, 'r') as ttm_raw_file:
        ttm_raw = ttm_raw_file.readlines()

    print("Read %d rows:" % (len(ttm_raw)))

    ttm_formatted = cleanup_and_format_data_set(ttm_raw)

    with open(ttm_formatted_path, 'wb') as ttm_formatted_file:
        wr = csv.writer(ttm_formatted_file, delimiter=' ')
        wr.writerows(ttm_formatted)

if __name__ == "__main__":
    main()
