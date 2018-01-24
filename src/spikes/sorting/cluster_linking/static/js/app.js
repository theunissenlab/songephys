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
  '$window',
  function($location, $http, $scope, $window) {
      $scope.TOSUBMIT = [];

      $http.get(
        'data'
      ).then(function(data) {
        $scope.timeBlocks = data.data;
        $scope.ONDECK = [];
        $scope.timeBlocks.forEach(function(timeBlock, i) {
            $scope.ONDECK.push([]);
        });
        return data;
      });
    
      $scope.isActive = function(blockIdx, cluster) {
          return $scope.ONDECK[blockIdx].indexOf(cluster) !== -1;
      };

      $scope.isSelected = function(blockIdx, cluster) {
          var result = false;
          $scope.TOSUBMIT.forEach(function(unit) {
              if (unit[blockIdx].indexOf(cluster) !== -1) {
                  result = true;
              }
          });
          return result
      };

      $scope.toggle = function(timeBlockIndex, clusterId) {
          if ($scope.isSelected(timeBlockIndex, clusterId)) {
              return;
          }
          const index = $scope.ONDECK[timeBlockIndex].indexOf(clusterId);
          if (index !== -1) {
              $scope.ONDECK[timeBlockIndex].splice(index, 1);
          } else {
              $scope.ONDECK[timeBlockIndex].push(clusterId)
          }
      };

      $scope.unravel = function(unitarr) {
          var result = [];
          unitarr.forEach(function(subarr) {
              result = result.concat(subarr);
          });
          return result;
      };

      $scope.clear = function() {
          $scope.ONDECK.forEach(function(timeBlockMembers) {
              timeBlockMembers.splice(0, timeBlockMembers.length);  // clear out the array
          });
      };

      $scope.save_new_cluster = function() {
          if ($scope.unravel($scope.ONDECK).length === 0) {
              return;
          }
          $scope.TOSUBMIT.push($scope.ONDECK);
          $scope.ONDECK = [];
          $scope.timeBlocks.forEach(function() {
              $scope.ONDECK.push([]);
          });
      };

      $scope.ungroup_cluster = function(clusterIndex) {
          $scope.TOSUBMIT.splice(clusterIndex, 1);
      };

      $scope.submit = function() {
          $http({
              url: '/submit',
              method: 'POST',
              data: JSON.stringify({
                  units: $scope.TOSUBMIT
              }),
              headers: {'Content-Type': 'application/json'}
          }).then(function() {
              $http.get('/shutdown').then(function() {
                  $window.close()
              });
          });
      };

  }
]);



