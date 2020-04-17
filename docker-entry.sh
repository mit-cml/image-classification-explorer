#!/bin/sh
mkdir /run/nginx
nginx
(cd /opt/backend; gunicorn --workers 2 --bind 0.0.0.0:5000 wsgi)



