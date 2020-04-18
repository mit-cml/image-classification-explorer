#!/bin/sh
mkdir /run/nginx
mkdir /opt/backend/temp
nginx
(cd /opt/backend; gunicorn --workers 2 --bind 0.0.0.0:5000 wsgi)



