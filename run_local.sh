# example

cd C:\Users\yusen.yang\Desktop\workspace\SGDBS.github.io

$nodeHome = "C:\Users\yusen.yang\Desktop\workspace\SGDBS.github.io\.fnm\node-versions\v24.16.0\installation"
$env:PATH = "$nodeHome;$env:PATH"

& "$nodeHome\npx.cmd" hexo clean
& "$nodeHome\npx.cmd" hexo generate
& "$nodeHome\npx.cmd" hexo server -p 4000