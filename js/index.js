//var myPolygon;
function initialize() {
  // Map Center
  var myLatLng = new google.maps.LatLng(10.83128,78.69476);
  // General Options
  var mapOptions = {
    zoom: 18,
    center: myLatLng,
    mapTypeId: 'satellite' //google.maps.MapTypeId.RoadMap
	
  };
 	var mapOptions = new MapLabel({
    text: 'Test',
    position: new google.maps.LatLng(10.83128,78.69476),
    map: map,
    fontSize: 35,
    align: 'right'
  });
  var map = new google.maps.Map(document.getElementById('map-canvas'),mapOptions);
  
  
  // Polygon Coordinates
  var triangleCoords = [
	new google.maps.LatLng(10.83128,78.69476), 
	new google.maps.LatLng(10.83186,78.69408), 
	new google.maps.LatLng(10.83187,78.6949)

  ];
  // Styling & Controls
  myPolygon = new google.maps.Polygon({
    paths: triangleCoords,
    draggable: true, // turn off if it gets annoying
    editable: true,
    strokeColor: '#FF0000',
    strokeOpacity: 0.8,
    strokeWeight: 2,
    fillColor: '#FF0000',
    fillOpacity: 0.35
  });

  myPolygon.setMap(map);
  //google.maps.event.addListener(myPolygon, "dragend", getPolygonCoords);
  google.maps.event.addListener(myPolygon.getPath(), "insert_at", getPolygonCoords);
  //google.maps.event.addListener(myPolygon.getPath(), "remove_at", getPolygonCoords);
  google.maps.event.addListener(myPolygon.getPath(), "set_at", getPolygonCoords);
}

//Display Coordinates below map
function getPolygonCoords() {
  var len = myPolygon.getPath().getLength();
  var htmlStr = "";
  for (var i = 0; i < len; i++) {
    htmlStr += "new google.maps.LatLng(" + myPolygon.getPath().getAt(i).toUrlValue(5) + "), ";
    //Use this one instead if you want to get rid of the wrap > new google.maps.LatLng(),
    //htmlStr += "" + myPolygon.getPath().getAt(i).toUrlValue(5);
  }
  document.getElementById('info').innerHTML = htmlStr;
}
function copyToClipboard(text) {
  window.prompt("Copy to clipboard: Ctrl+C, Enter", text);
}