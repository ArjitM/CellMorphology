
dir = getDirectory("image");
name = getTitle();
//b = split(name, "-RFP");
//c = b[0];
//a = replace(replace(c, "-", ""), " ","");
//place = dir +a[0]+a[1]+"_RFPsequence";
a = replace(name, "-RFP.tif", "");
place = dir + a + "_RFPsequence";
File.makeDirectory(place);
run("Image Sequence... ", "format=TIF name=piece- start=1 save=" + place);