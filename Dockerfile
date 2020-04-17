FROM 127.0.0.1:5000/alpine
MAINTAINER Jeffrey I. Schiller <jis@mit.edu>

RUN apk add py3-gunicorn py3-setuptools py3-numpy nginx npm git freetype freetype-dev gcc musl-dev g++ \
            python3-dev py3-flask py-numpy-dev py3-scipy py3-requests

ADD c1.conf /etc/nginx/conf.d/default.conf
ADD docker-entry.sh /docker-entry.sh
RUN chmod 755 /docker-entry.sh

RUN easy_install-3.6 cython
RUN easy_install-3.6 matplotlib

RUN easy_install-3.6 flask-cors
RUN easy_install-3.6 pydub

ADD frontend /opt/frontend
ADD backend /opt/backend
ADD ssl /etc/ssl

CMD /docker-entry.sh

