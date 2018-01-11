angular.module('ClusterLinking', [
  'ngRoute',
  'controllers'
])

.config([
  '$routeProvider',
  function($routeProvider) {
    $routeProvider
      .when('/', {
        controller: 'home',
        templateUrl: 'home.html'
      });
    $routeProvider.otherwise({redirectTo: '/'});
  }
]);

angular.module('controllers', [])

.controller('home', [
  '$location',
  '$http',
  '$scope',
  function($location, $http, $scope) {
      // load up all data for electrode and unit
      // send all figures down to client
      // send metrics down to clinet
      //
      // send a set of units up to be grouped
      // 
      //
      //
      $scope.timeBlocks = [
          {
              t_start: 0,
              t_stop: 600,
              scatter: "static/img/scatter.png",
              clusters: {
                  0: {
                      waveforms: "static/img/spike_0.png",
                      selected: false,
                      active: false
                  },
                  2: {
                      waveforms: "static/img/spike_2.png",
                      selected: false,
                      active: false
                  },
                  1: {
                      waveforms: "static/img/spike_1.png",
                      selected: false,
                      active: false
                  },
              }
          },
          {
              t_start: 300,
              t_stop: 900,
              scatter: "static/img/scatter.png",
              clusters: {
                  2: {
                      waveforms: "static/img/spike_0.png",
                      selected: false,
                      active: false
                  },
                  3: {
                      waveforms: "static/img/spike_2.png",
                      selected: false,
                      active: false
                  },
                  4: {
                      waveforms: "static/img/spike_1.png",
                      selected: false,
                      active: false
                  },
              }
          }
      ];

      $http.get(
        'data'
      ).then(function(data) {
        $scope.timeBlocks = data.data;
        $scope.timeBlocks.forEach(function(timeBlock, i) {
            Object.keys(timeBlock.clusters).forEach(function(c) {
                timeBlock.clusters[c].block = i;
            })
        });
        return data;
      });
    
      var nextId = 0;

      // When we select a new unit, we will add it to this set
      $scope.newUnits = [];

      // list of [timeblock, idx] pairs
      $scope.ondeck = {};
      $scope.toggle = function(timeBlock, cluster, id) {
          if (!!cluster.selected) {
              return;
          }

          if (!!$scope.ondeck[[timeBlock.t_start, id]]) {
              cluster.active = false;
              delete $scope.ondeck[[timeBlock.t_start, id]];
          } else {
              cluster.active = true;
              $scope.ondeck[[timeBlock.t_start, id]] = cluster;
          }
          console.log($scope.ondeck);
      };
      $scope.clear = function() {
          Object.keys($scope.ondeck).forEach(function(key) {
              $scope.ondeck[key].active = false;
          });
          $scope.ondeck = {};
      };

      $scope.save_new_cluster = function() {
          if (!Object.keys($scope.ondeck).length) {
              console.log("warning... nothing selecetd");
              return;
          }
          var newUnit = {
              id: nextId,
              pieces: $scope.ondeck
          }
          Object.keys($scope.ondeck).forEach(function(key) {
              $scope.ondeck[key].active = false;
              $scope.ondeck[key].selected = true;
          });
          $scope.newUnits.push(newUnit);
          $scope.ondeck = {};
          nextId += 1;
      };

      $scope.ungroup_cluster = function(id) {
          var removedUnits = $scope.newUnits.filter(
              unit => unit.id === id
          );
          $scope.newUnits = $scope.newUnits.filter(
              unit => unit.id !== id
          );
          Object.keys(removedUnits).forEach(function(key) {
              Object.keys(removedUnits[key].pieces).forEach(function(k) {
                  removedUnits[key].pieces[k].active = false;
                  removedUnits[key].pieces[k].selected = false;
              });
          });
      };

      var submit = function() {
          var result = [];
          $scope.newUnits.forEach(function(unit) {
            var pieces = [];
            Object.keys(unit.pieces).forEach(function(piece) {
                pieces.push([unit.pieces[piece].block, piece])
            });
            result.push({
                id: unit.id,
                pieces: pieces
            });
          });
          return result;
      };

      $scope.submit = function() {
          var result = submit();
          console.log(result);
          $http({
              url: '/submit',
              method: 'POST',
              data: JSON.stringify({
                  units: result
              }),
              headers: {'Content-Type': 'application/json'}
          });
      };

  }
]);



