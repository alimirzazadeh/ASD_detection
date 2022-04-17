#!C:\Users\alimi\AppData\Local\Programs\Python\Python39
import cgi,os
print('content-type:text/html\r\n\r\n')

form=cgi.FieldStorage()
pn = int(form.getvalue("q1"))
print('<html>')
print('<body>')
print('<h1> ' + pn + ' </h1>')
print('</body></html>')