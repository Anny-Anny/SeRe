import yagmail

#链接邮箱服务器
yag = yagmail.SMTP(user="1207327296@qq.com", password="dhslifceafwahjia", host='smtp.qq.com')

#邮箱正文
contents = ['This is the body, and here is just text http://somedomain/image.png',
            'You can find an audio file attached.', '/local/path/song.mp3']

# 发送邮件
yag.send('1207327296@qq.com', 'subject', contents)