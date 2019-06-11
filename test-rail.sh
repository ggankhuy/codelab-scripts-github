# full instructions for centos1811 available at: https://www.howtoforge.com/tutorial/centos-lamp-server-apache-mysql-php/

SLEEP_TIME = 3
rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY*
yum -y install epel-release
yum -y install mariadb-server mariadb
systemctl start mariadb.service
systemctl enable mariadb.service
mysql_secure_installation
yum -y install httpd
systemctl start httpd.service
systemctl enable httpd.service

firewall-cmd --permanent --zone=public --add-service=http 
firewall-cmd --permanent --zone=public --add-service=https
firewall-cmd --reload

rpm -Uvh http://rpms.remirepo.net/enterprise/remi-release-7.rpm
yum -y install yum-utils
yum update

yum-config-manager --enable remi-php72
yum -y install php php-opcache

systemctl restart httpd.service

# copy php.info to /var/www/html/info.php

# curl http://localhost/info.php
sleep $SLEEP_TIME

yum -y install php-mysqlnd php-pdo
yum -y install php-gd php-ldap php-odbc php-pear php-xml php-xmlrpc php-mbstring php-soap curl curl-devel

systemctl restart httpd.service

yum -y install phpMyAdmin

# nano /etc/httpd/conf.d/phpMyAdmin.conf
# insert into phpMyAdmin.conf:  
#[...]
#$cfg['Servers'][$i]['auth_type']     = 'http';    // Authentication method (config, http or cookie based)?
#[...]

