
curl -o buildAgentFull.zip https://g900.teamcity.com/update/buildAgentFull.zip

unzip buildAgentFull.zip -d buildAgent

mv buildAgent/conf/buildAgent.dist.properties buildAgent/conf/buildAgent.properties

sed -i '' 's|^name=.*|name=amd_27|' buildAgent/conf/buildAgent.properties

sed -i '' 's|^authorizationToken=.*|authorizationToken=singleUseAutomaticAgentAuthorizationToken:Qn7vSzxPM22a485acSijWkCPp4p2v0O3|' buildAgent/conf/buildAgent.properties

sed -i '' 's|^serverUrl=.*|serverUrl=https://g900.teamcity.com|' buildAgent/conf/buildAgent.properties

chmod +x buildAgent/bin/agent.sh

buildAgent/bin/agent.sh start
