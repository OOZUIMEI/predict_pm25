import xlrd

wb = xlrd.open_workbook("test_aws.xls")

print(wb.sheet_names())