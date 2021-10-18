var initial_offset = parseInt(Args.offset);
var members = API.groups.getMembers({"group_id": Args.group_id, "v": "5.27", "sort": "id_asc", "count": "1000", "offset": initial_offset}).items; // making first request and creating array
var info = API.users.get({"user_ids": members, "fields": "city,country,personal,sex,bdate", "name_case": "Nom"});
var offset = 1000; // shift for group users
while (offset < 10000 && (offset + initial_offset) < Args.total_count) // while didn't iterate by 20000 users or didn't reach group end
{
  var new_members = API.groups.getMembers({"group_id": Args.group_id, "v": "5.27", "sort": "id_asc", "count": "1000", "offset": (initial_offset + offset)}).items;
  info = info + API.users.get({"user_ids": new_members, "fields": "city,country,personal,sex,bdate", "name_case": "Nom"});
  offset = offset + 1000; // increase shift by 1000
};
return info;