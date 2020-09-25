create table categories (category text, minnutrition real, maxnutrition real); 
insert into categories values('calories',1800,2200);
insert into categories values('protein',91,1e100);
insert into categories values('fat',0,65);
insert into categories values('sodium',0,1779);

create table foods (food text, cost real);
insert into foods values('hamburger',2.49);
insert into foods values('chicken',2.89);
insert into foods values('hot dog',1.5);
insert into foods values('fries',1.89);
insert into foods values('macaroni',2.09);
insert into foods values('pizza',1.99);
insert into foods values('salad',2.49);
insert into foods values('milk',0.89);
insert into foods values('ice cream',1.59);

create table nutrition (food text, category text, value real);
insert into nutrition values('ice cream','protein',8);
insert into nutrition values('macaroni','protein',12);
insert into nutrition values('fries','sodium',270);
insert into nutrition values('fries','calories',380);
insert into nutrition values('hamburger','fat',26);
insert into nutrition values('macaroni','sodium',930);
insert into nutrition values('hot dog','sodium',1800);
insert into nutrition values('chicken','sodium',1190);
insert into nutrition values('salad','calories',320);
insert into nutrition values('ice cream','calories',330);
insert into nutrition values('milk','sodium',125);
insert into nutrition values('salad','sodium',1230);
insert into nutrition values('pizza','sodium',820);
insert into nutrition values('ice cream','fat',10);
insert into nutrition values('pizza','protein',15);
insert into nutrition values('pizza','calories',320);
insert into nutrition values('hamburger','calories',410);
insert into nutrition values('milk','fat',2.5);
insert into nutrition values('salad','protein',31);
insert into nutrition values('milk','protein',8);
insert into nutrition values('hot dog','protein',20);
insert into nutrition values('salad','fat',12);
insert into nutrition values('hot dog','fat',32);
insert into nutrition values('chicken','fat',10);
insert into nutrition values('chicken','protein',32);
insert into nutrition values('fries','protein',4);
insert into nutrition values('pizza','fat',12);
insert into nutrition values('milk','calories',100);
insert into nutrition values('ice cream','sodium',180);
insert into nutrition values('chicken','calories',420);
insert into nutrition values('hamburger','sodium',730);
insert into nutrition values('macaroni','calories',320);
insert into nutrition values('fries','fat',19);
insert into nutrition values('hot dog','calories',560);
insert into nutrition values('macaroni','fat',10);
insert into nutrition values('hamburger','protein',24);
